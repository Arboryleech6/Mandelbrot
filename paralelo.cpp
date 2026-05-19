/**
 * mandelbrot_fractal.cpp
 * ============================================================
 * Programa paralelo con OpenMP en C++17 para Linux
 *
 * Tarea A: Genera imagen 16K (15360x8640) del conjunto de Mandelbrot
 *          con coloración suavizada (smooth coloring).
 *          Paralelizado con OpenMP — benchmarking de planificadores.
 *
 * Tarea B: Aplica filtro Gaussiano separable (radio=63, kernel 127x127)
 *          y filtro Sobel de detección de bordes.
 *          Paralelizado con OpenMP (schedule static por fila).
 *
 * ============================================================
 * COMPILACIÓN
 * ============================================================
 *   g++ -std=c++17 -O2 -fopenmp -o mandelbrot mandelbrot_fractal.cpp
 *
 *   Para controlar el número de hilos en tiempo de ejecución:
 *   OMP_NUM_THREADS=8 ./mandelbrot
 *
 * ============================================================
 * GUÍA DE PLANIFICADORES — TAREA A (renderMandelbrot)
 * ============================================================
 *
 * El Mandelbrot tiene carga MUY DESIGUAL por fila: las filas que
 * atraviesan el interior del conjunto son baratas (se cortan pronto
 * con la prueba cardioide), mientras que las del borde requieren
 * MAX_ITER iteraciones completas. Esto hace que el planificador
 * importe mucho.
 *
 * Los tres planificadores disponibles se controlan cambiando la
 * macro SCHED_MANDELBROT justo abajo:
 *
 *  ┌─────────────┬──────────────────────────────────────────────────────┐
 *  │ Valor macro │ Directiva OpenMP resultante                          │
 *  ├─────────────┼──────────────────────────────────────────────────────┤
 *  │ STATIC      │ schedule(static)          — reparto estático igual   │
 *  │ DYNAMIC_N   │ schedule(dynamic, CHUNK)  — reparto dinámico         │
 *  │ GUIDED_N    │ schedule(guided,  CHUNK)  — reparto guiado           │
 *  └─────────────┴──────────────────────────────────────────────────────┘
 *
 *  CHUNK_SIZE controla el tamaño de bloque para dynamic y guided.
 *
 * ── ¿Cómo hacer el benchmark empírico? ──────────────────────────────
 *
 *  1. Compila una vez con el planificador deseado y mide con `time`:
 *
 *       # Paso 1 — Static (línea base, sin modificar nada)
 *       g++ -std=c++17 -O2 -fopenmp -o mandelbrot mandelbrot_fractal.cpp
 *       time ./mandelbrot
 *
 *       # Paso 2 — Dynamic, chunk=1
 *       #   Edita: #define SCHED_MANDELBROT DYNAMIC_N
 *       #          #define CHUNK_SIZE 1
 *       g++ -std=c++17 -O2 -fopenmp -o mandelbrot mandelbrot_fractal.cpp
 *       time ./mandelbrot
 *
 *       # Paso 3 — Dynamic, chunk=4  (edita CHUNK_SIZE 4)
 *       # Paso 4 — Dynamic, chunk=16
 *       # Paso 5 — Dynamic, chunk=32
 *       # Paso 6 — Guided,  chunk=1  (edita SCHED_MANDELBROT GUIDED_N)
 *       # Paso 7 — Guided,  chunk=4
 *       # Paso 8 — Guided,  chunk=16
 *
 *  2. El programa imprime el tiempo exacto de Tarea A en la línea:
 *       [Tarea A] Listo en X.XX s
 *     Usa ese valor para comparar entre configuraciones.
 *
 *  Valores sugeridos de CHUNK_SIZE para explorar:
 *    dynamic : 1, 2, 4, 8, 16, 32, 64
 *    guided  : 1, 4, 8, 16
 *
 * ── Intuición de cada planificador ──────────────────────────────────
 *
 *  STATIC   — Divide las IMG_H filas en bloques iguales y fijos antes
 *             de empezar. Mínima sincronización, pero si unas filas
 *             son más lentas que otras el hilo más cargado marca el
 *             tiempo total (desbalance).
 *
 *  DYNAMIC  — Los hilos piden CHUNK_SIZE filas cuando quedan libres.
 *             Balancea la carga variable pero genera más overhead de
 *             sincronización (mutex interno en el runtime de OpenMP).
 *             Chunks pequeños → mejor balance, más overhead.
 *             Chunks grandes → menos overhead, peor balance.
 *
 *  GUIDED   — Como dynamic pero el chunk empieza grande
 *             (≈ filas_restantes / num_hilos) y decrece hasta CHUNK_SIZE
 *             mínimo. Combina baja sincronización al inicio con buen
 *             balance al final donde la varianza de carga es mayor.
 *
 * ============================================================
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <string>
#include <algorithm>
#include <cstring>
#include <omp.h>

// ════════════════════════════════════════════════════════════════════════════
// CONFIGURACIÓN DE PLANIFICADOR — MODIFICA AQUÍ PARA EL BENCHMARK
// ════════════════════════════════════════════════════════════════════════════

// Elige el planificador para Tarea A descomentando UNA sola línea:
//#define SCHED_MANDELBROT STATIC       // ← planificador por defecto (línea base)
#define SCHED_MANDELBROT DYNAMIC_N
// #define SCHED_MANDELBROT GUIDED_N

// Tamaño de bloque para dynamic y guided (ignorado con STATIC):
#define CHUNK_SIZE 2

// ─── Valores internos — NO modificar ─────────────────────────────────────────
#define STATIC    0
#define DYNAMIC_N 1
#define GUIDED_N  2

// Helper para convertir el valor de la macro a string en mensajes
#define STRINGIFY2(x) #x
#define STRINGIFY(x)  STRINGIFY2(x)

// ─── Parámetros globales ──────────────────────────────────────────────────────
static constexpr int    IMG_W        = 15360;
static constexpr int    IMG_H        = 8640;
static constexpr int    MAX_ITER     = 4096;   // iteraciones Mandelbrot (mayor detalle y carga)

static constexpr double CENTER_X     = -0.75;
static constexpr double CENTER_Y     =  0.0;
static constexpr double ZOOM         =  2.35;  // mitad del rango en Re

static constexpr int    GAUSS_RADIUS = 63;     // radio del kernel Gaussiano — filtro pesado
static constexpr double GAUSS_SIGMA  = 15.0;

// ─── Tipo Pixel ───────────────────────────────────────────────────────────────
struct Pixel { uint8_t r, g, b; };
using Image = std::vector<Pixel>;

// ─── Utilidades de tiempo ─────────────────────────────────────────────────────
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ─── Nombre del planificador activo (para mensajes en consola) ────────────────
static const char* schedName() {
#if   SCHED_MANDELBROT == DYNAMIC_N
    return "dynamic(" STRINGIFY(CHUNK_SIZE) ")";
#elif SCHED_MANDELBROT == GUIDED_N
    return "guided(" STRINGIFY(CHUNK_SIZE) ")";
#else
    return "static";
#endif
}

// ─── Barra de progreso (segura con OpenMP: solo hilo 0 imprime) ───────────────
static void progress(const char* tag, int cur, int total) {
    if (omp_get_thread_num() != 0) return;   // solo el hilo master imprime
    if (cur % 250 != 0 && cur != total - 1) return;
    int bar = static_cast<int>(40.0 * cur / total);
    std::cout << "\r" << tag << " [";
    for (int i = 0; i < 40; ++i) std::cout << (i < bar ? '#' : '.');
    std::cout << "] " << static_cast<int>(100.0 * cur / total) << "%  ";
    std::cout.flush();
}

// ─── Paleta de color (suavizada) ──────────────────────────────────────────────
static Pixel palette(double t) {
    // negro → índigo → cian eléctrico → naranja → blanco
    static const double P[][3] = {
        {0.00, 0.00, 0.00},
        {0.08, 0.04, 0.48},
        {0.00, 0.58, 0.98},
        {1.00, 0.60, 0.00},
        {1.00, 1.00, 1.00},
    };
    t = std::clamp(t, 0.0, 1.0) * 4.0;
    int    i = std::min(static_cast<int>(t), 3);
    double f = t - i;
    auto lerp = [&](int ch) -> uint8_t {
        return static_cast<uint8_t>(
            std::clamp(P[i][ch]*(1-f) + P[i+1][ch]*f, 0.0, 1.0) * 255.0);
    };
    return {lerp(0), lerp(1), lerp(2)};
}

// ════════════════════════════════════════════════════════════════════════════
// TAREA A — Renderizado Mandelbrot (paralelizado con OpenMP)
// ════════════════════════════════════════════════════════════════════════════
//
// El bucle externo (filas) se paralela con #pragma omp parallel for.
// El planificador se selecciona en tiempo de compilación mediante las macros
// SCHED_MANDELBROT y CHUNK_SIZE definidas arriba.
//
// Cada píxel es independiente → no hay condiciones de carrera.
// La barra de progreso solo la actualiza el hilo con id 0.
//
Image renderMandelbrot() {
    Image img(IMG_W * IMG_H);
    const double scale = (2.0 * ZOOM) / std::min(IMG_W, IMG_H);

    std::cout << "\n[Tarea A] Mandelbrot " << IMG_W << "x" << IMG_H
              << "  max_iter=" << MAX_ITER
              << "  hilos=" << omp_get_max_threads()
              << "  sched=" << schedName() << "\n";
    auto t0 = Clock::now();

    // ── Selección de planificador en tiempo de compilación ────────────────
    // Solo se activa la rama correspondiente a SCHED_MANDELBROT.
    // Las otras ramas son eliminadas por el preprocesador.

#if SCHED_MANDELBROT == DYNAMIC_N
    // ── dynamic: cada hilo pide CHUNK_SIZE filas al quedar libre ──────────
    #pragma omp parallel for schedule(dynamic, CHUNK_SIZE) default(none) \
            shared(img) firstprivate(scale)
    for (int py = 0; py < IMG_H; ++py) {
        progress("  Render ", py, IMG_H);
        const double cy = CENTER_Y + (py - IMG_H * 0.5) * scale;
        for (int px = 0; px < IMG_W; ++px) {
            const double cx = CENTER_X + (px - IMG_W * 0.5) * scale;
            {
                double p = std::sqrt((cx - 0.25)*(cx - 0.25) + cy*cy);
                if (cx < p - 2*p*p + 0.25 ||
                    (cx+1)*(cx+1) + cy*cy < 0.0625) {
                    img[py * IMG_W + px] = {0, 0, 0};
                    continue;
                }
            }
            double zx = 0.0, zy = 0.0; int n = 0;
            while (zx*zx + zy*zy <= 4.0 && n < MAX_ITER) {
                double t = zx*zx - zy*zy + cx;
                zy = 2.0*zx*zy + cy; zx = t; ++n;
            }
            if (n == MAX_ITER) {
                img[py * IMG_W + px] = {0, 0, 0};
            } else {
                double mod    = std::sqrt(zx*zx + zy*zy);
                double smooth = n - std::log2(std::log2(mod));
                double t      = std::fmod(smooth / static_cast<double>(MAX_ITER) * 5.0, 1.0);
                img[py * IMG_W + px] = palette(t);
            }
        }
    }

#elif SCHED_MANDELBROT == GUIDED_N
    // ── guided: chunk decrece desde (filas/hilos) hasta CHUNK_SIZE mínimo ─
    #pragma omp parallel for schedule(guided, CHUNK_SIZE) default(none) \
            shared(img) firstprivate(scale)
    for (int py = 0; py < IMG_H; ++py) {
        progress("  Render ", py, IMG_H);
        const double cy = CENTER_Y + (py - IMG_H * 0.5) * scale;
        for (int px = 0; px < IMG_W; ++px) {
            const double cx = CENTER_X + (px - IMG_W * 0.5) * scale;
            {
                double p = std::sqrt((cx - 0.25)*(cx - 0.25) + cy*cy);
                if (cx < p - 2*p*p + 0.25 ||
                    (cx+1)*(cx+1) + cy*cy < 0.0625) {
                    img[py * IMG_W + px] = {0, 0, 0};
                    continue;
                }
            }
            double zx = 0.0, zy = 0.0; int n = 0;
            while (zx*zx + zy*zy <= 4.0 && n < MAX_ITER) {
                double t = zx*zx - zy*zy + cx;
                zy = 2.0*zx*zy + cy; zx = t; ++n;
            }
            if (n == MAX_ITER) {
                img[py * IMG_W + px] = {0, 0, 0};
            } else {
                double mod    = std::sqrt(zx*zx + zy*zy);
                double smooth = n - std::log2(std::log2(mod));
                double t      = std::fmod(smooth / static_cast<double>(MAX_ITER) * 5.0, 1.0);
                img[py * IMG_W + px] = palette(t);
            }
        }
    }

#else // STATIC — planificador por defecto
    // ── static: reparto en bloques iguales de (IMG_H / num_hilos) filas ───
    #pragma omp parallel for schedule(static) default(none) \
            shared(img) firstprivate(scale)
    for (int py = 0; py < IMG_H; ++py) {
        progress("  Render ", py, IMG_H);
        const double cy = CENTER_Y + (py - IMG_H * 0.5) * scale;
        for (int px = 0; px < IMG_W; ++px) {
            const double cx = CENTER_X + (px - IMG_W * 0.5) * scale;
            {
                double p = std::sqrt((cx - 0.25)*(cx - 0.25) + cy*cy);
                if (cx < p - 2*p*p + 0.25 ||
                    (cx+1)*(cx+1) + cy*cy < 0.0625) {
                    img[py * IMG_W + px] = {0, 0, 0};
                    continue;
                }
            }
            double zx = 0.0, zy = 0.0; int n = 0;
            while (zx*zx + zy*zy <= 4.0 && n < MAX_ITER) {
                double t = zx*zx - zy*zy + cx;
                zy = 2.0*zx*zy + cy; zx = t; ++n;
            }
            if (n == MAX_ITER) {
                img[py * IMG_W + px] = {0, 0, 0};
            } else {
                double mod    = std::sqrt(zx*zx + zy*zy);
                double smooth = n - std::log2(std::log2(mod));
                double t      = std::fmod(smooth / static_cast<double>(MAX_ITER) * 5.0, 1.0);
                img[py * IMG_W + px] = palette(t);
            }
        }
    }
#endif

    std::cout << "\n[Tarea A] Listo en " << elapsed(t0) << " s\n";
    return img;
}

// ════════════════════════════════════════════════════════════════════════════
// PNG mínimo sin dependencias externas
// Usa deflate "stored" (sin compresión real) — siempre correcto en Linux.
// El archivo PNG resultante es más grande que un PNG comprimido pero
// perfectamente válido y legible por cualquier visor/editor.
// ════════════════════════════════════════════════════════════════════════════

// CRC32
static uint32_t CRC_TABLE[256];
static void initCRC() {
    for (uint32_t n = 0; n < 256; ++n) {
        uint32_t c = n;
        for (int k = 0; k < 8; ++k)
            c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        CRC_TABLE[n] = c;
    }
}
static uint32_t crc32(const uint8_t* d, size_t len, uint32_t crc = 0xFFFFFFFFu) {
    for (size_t i = 0; i < len; ++i)
        crc = CRC_TABLE[(crc ^ d[i]) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

// Adler-32
static uint32_t adler32(const uint8_t* d, size_t len) {
    uint32_t s1 = 1, s2 = 0;
    for (size_t i = 0; i < len; ++i) {
        s1 = (s1 + d[i]) % 65521u;
        s2 = (s2 + s1)   % 65521u;
    }
    return (s2 << 16) | s1;
}

static void u32be(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back((x>>24)&0xFF); v.push_back((x>>16)&0xFF);
    v.push_back((x>> 8)&0xFF); v.push_back( x     &0xFF);
}

// Wrapper de chunk PNG
static void writeChunk(std::vector<uint8_t>& file,
                       const char type[4],
                       const std::vector<uint8_t>& data) {
    u32be(file, static_cast<uint32_t>(data.size()));
    file.push_back(type[0]); file.push_back(type[1]);
    file.push_back(type[2]); file.push_back(type[3]);
    file.insert(file.end(), data.begin(), data.end());
    std::vector<uint8_t> td;
    td.reserve(4 + data.size());
    td.push_back(type[0]); td.push_back(type[1]);
    td.push_back(type[2]); td.push_back(type[3]);
    td.insert(td.end(), data.begin(), data.end());
    u32be(file, crc32(td.data(), td.size()));
}

// Deflate stored (BTYPE=00, sin compresión real)
static std::vector<uint8_t> deflateStored(const uint8_t* src, size_t len) {
    std::vector<uint8_t> out;
    out.reserve(len + len/65535 * 5 + 10);
    out.push_back(0x78); out.push_back(0x01);
    size_t pos = 0;
    while (pos < len) {
        size_t   blk  = std::min<size_t>(len - pos, 65535);
        bool     last = (pos + blk >= len);
        uint16_t bl   = static_cast<uint16_t>(blk);
        out.push_back(last ? 0x01 : 0x00);
        out.push_back( bl        & 0xFF);
        out.push_back((bl >>  8) & 0xFF);
        out.push_back((~bl)      & 0xFF);
        out.push_back((~bl >> 8) & 0xFF);
        out.insert(out.end(), src + pos, src + pos + blk);
        pos += blk;
    }
    uint32_t a = adler32(src, len);
    out.push_back((a>>24)&0xFF); out.push_back((a>>16)&0xFF);
    out.push_back((a>> 8)&0xFF); out.push_back( a     &0xFF);
    return out;
}

static void savePNG(const std::string& filename, const Image& img, int w, int h) {
    std::cout << "  Guardando PNG (" << w << "x" << h << "): " << filename << "  ";
    std::cout.flush();
    auto t0 = Clock::now();

    std::vector<uint8_t> raw;
    raw.reserve(static_cast<size_t>(h) * (1 + w * 3));
    for (int y = 0; y < h; ++y) {
        raw.push_back(0x00);
        for (int x = 0; x < w; ++x) {
            raw.push_back(img[y*w+x].r);
            raw.push_back(img[y*w+x].g);
            raw.push_back(img[y*w+x].b);
        }
    }

    auto idat = deflateStored(raw.data(), raw.size());

    std::vector<uint8_t> file;
    initCRC();

    static const uint8_t sig[] = {137,80,78,71,13,10,26,10};
    file.insert(file.end(), sig, sig+8);

    std::vector<uint8_t> ihdr(13,0);
    uint32_t wu=w, hu=h;
    ihdr[0]=(wu>>24)&0xFF; ihdr[1]=(wu>>16)&0xFF; ihdr[2]=(wu>>8)&0xFF; ihdr[3]=wu&0xFF;
    ihdr[4]=(hu>>24)&0xFF; ihdr[5]=(hu>>16)&0xFF; ihdr[6]=(hu>>8)&0xFF; ihdr[7]=hu&0xFF;
    ihdr[8]=8; ihdr[9]=2;
    writeChunk(file, "IHDR", ihdr);
    writeChunk(file, "IDAT", idat);
    writeChunk(file, "IEND", {});

    std::ofstream ofs(filename, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(file.data()),
              static_cast<std::streamsize>(file.size()));

    std::cout << "OK  (" << file.size()/1024/1024 << " MB, "
              << elapsed(t0) << " s)\n";
}

// ════════════════════════════════════════════════════════════════════════════
// TAREA B — Filtros de convolución 2D (paralelizados con OpenMP, schedule static)
// ════════════════════════════════════════════════════════════════════════════
//
// La convolución Gaussiana y el filtro Sobel se paralelizan por filas con
// schedule(static): la carga es uniforme (todos los píxeles hacen el mismo
// trabajo), así que static es el planificador óptimo aquí.
//

struct ImageF {
    std::vector<float> r, g, b;
    int w, h;
    ImageF(int w, int h) : r(w*h,0.f), g(w*h,0.f), b(w*h,0.f), w(w), h(h) {}
};

static ImageF toFloat(const Image& img, int w, int h) {
    ImageF f(w,h);
    // Conversión embarazosamente paralela — sin dependencias entre píxeles
    #pragma omp parallel for schedule(static) default(none) shared(f, img) firstprivate(w, h)
    for (int i = 0; i < w*h; ++i) {
        f.r[i]=img[i].r/255.f; f.g[i]=img[i].g/255.f; f.b[i]=img[i].b/255.f;
    }
    return f;
}

static Image toUint8(const ImageF& f) {
    Image img(f.w*f.h);
    const int total = f.w * f.h;
    #pragma omp parallel for schedule(static) default(none) shared(f, img) firstprivate(total)
    for (int i = 0; i < total; ++i) {
        img[i].r=static_cast<uint8_t>(std::clamp(f.r[i],0.f,1.f)*255.f);
        img[i].g=static_cast<uint8_t>(std::clamp(f.g[i],0.f,1.f)*255.f);
        img[i].b=static_cast<uint8_t>(std::clamp(f.b[i],0.f,1.f)*255.f);
    }
    return img;
}

static std::vector<float> gaussKernel1D(int r, double sigma) {
    int sz=2*r+1; std::vector<float> k(sz); float sum=0;
    for (int i=0;i<sz;++i){double x=i-r; k[i]=static_cast<float>(std::exp(-(x*x)/(2*sigma*sigma))); sum+=k[i];}
    for (auto& v:k) v/=sum;
    return k;
}

// Convolución horizontal — paralelizada por filas (sin dependencia entre filas)
static void convH(const ImageF& src, ImageF& dst,
                  const std::vector<float>& k, int r) {
    #pragma omp parallel for schedule(static) default(none) \
            shared(src, dst, k) firstprivate(r)
    for (int py=0;py<src.h;++py) {
        if (omp_get_thread_num() == 0) progress("  GaussH ", py, src.h);
        for (int px=0;px<src.w;++px) {
            float vr=0,vg=0,vb=0;
            for (int dx=-r;dx<=r;++dx) {
                int sx=std::clamp(px+dx,0,src.w-1);
                float w=k[dx+r];
                vr+=w*src.r[py*src.w+sx];
                vg+=w*src.g[py*src.w+sx];
                vb+=w*src.b[py*src.w+sx];
            }
            dst.r[py*src.w+px]=vr;
            dst.g[py*src.w+px]=vg;
            dst.b[py*src.w+px]=vb;
        }
    }
    std::cout << "\n";
}

// Convolución vertical — paralelizada por filas (sin dependencia entre filas)
static void convV(const ImageF& src, ImageF& dst,
                  const std::vector<float>& k, int r) {
    #pragma omp parallel for schedule(static) default(none) \
            shared(src, dst, k) firstprivate(r)
    for (int py=0;py<src.h;++py) {
        if (omp_get_thread_num() == 0) progress("  GaussV ", py, src.h);
        for (int px=0;px<src.w;++px) {
            float vr=0,vg=0,vb=0;
            for (int dy=-r;dy<=r;++dy) {
                int sy=std::clamp(py+dy,0,src.h-1);
                float w=k[dy+r];
                vr+=w*src.r[sy*src.w+px];
                vg+=w*src.g[sy*src.w+px];
                vb+=w*src.b[sy*src.w+px];
            }
            dst.r[py*src.w+px]=vr;
            dst.g[py*src.w+px]=vg;
            dst.b[py*src.w+px]=vb;
        }
    }
    std::cout << "\n";
}

// Filtro Sobel — paralelizado por filas (sin dependencia entre filas de salida)
static Image sobelFilter(const ImageF& src) {
    int w=src.w, h=src.h;
    Image out(w*h);
    static const float Kx[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
    static const float Ky[3][3]={{-1,-2,-1},{0,0,0},{1,2,1}};

    #pragma omp parallel for schedule(static) default(none) \
            shared(src, out, Kx, Ky) firstprivate(w, h)
    for (int py=0;py<h;++py) {
        if (omp_get_thread_num() == 0) progress("  Sobel  ", py, h);
        for (int px=0;px<w;++px) {
            float gxR=0,gyR=0,gxG=0,gyG=0,gxB=0,gyB=0;
            for (int dy=-1;dy<=1;++dy) {
                int sy=std::clamp(py+dy,0,h-1);
                for (int dx=-1;dx<=1;++dx) {
                    int sx=std::clamp(px+dx,0,w-1);
                    float kx=Kx[dy+1][dx+1], ky=Ky[dy+1][dx+1];
                    int idx=sy*w+sx;
                    gxR+=kx*src.r[idx]; gyR+=ky*src.r[idx];
                    gxG+=kx*src.g[idx]; gyG+=ky*src.g[idx];
                    gxB+=kx*src.b[idx]; gyB+=ky*src.b[idx];
                }
            }
            auto mag=[](float a,float b){return std::clamp(std::sqrt(a*a+b*b)/4.f,0.f,1.f);};
            out[py*w+px]={
                static_cast<uint8_t>(mag(gxR,gyR)*255),
                static_cast<uint8_t>(mag(gxG,gyG)*255),
                static_cast<uint8_t>(mag(gxB,gyB)*255)
            };
        }
    }
    std::cout << "\n";
    return out;
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "============================================\n"
              << "  Mandelbrot 16K + Filtros  (OpenMP)\n"
              << "  Hilos disponibles : " << omp_get_max_threads() << "\n"
              << "  Planificador A    : " << schedName() << "\n"
              << "============================================\n";
    auto T0 = Clock::now();

    // ── Tarea A ──────────────────────────────────────────────────────────────
    Image fractal = renderMandelbrot();
    savePNG("mandelbrot_16k.png", fractal, IMG_W, IMG_H);

    // ── Tarea B ──────────────────────────────────────────────────────────────
    std::cout << "\n[Tarea B] Gaussiano separable (radio=" << GAUSS_RADIUS
              << ", sigma=" << GAUSS_SIGMA << ")  + Sobel"
              << "  hilos=" << omp_get_max_threads() << "\n";
    auto t1 = Clock::now();

    auto k1d = gaussKernel1D(GAUSS_RADIUS, GAUSS_SIGMA);
    ImageF src  = toFloat(fractal, IMG_W, IMG_H);
    ImageF tmp(IMG_W, IMG_H), blurF(IMG_W, IMG_H);

    convH(src,  tmp,   k1d, GAUSS_RADIUS);
    convV(tmp,  blurF, k1d, GAUSS_RADIUS);

    Image blurred = toUint8(blurF);
    savePNG("mandelbrot_16k_blur.png", blurred, IMG_W, IMG_H);

    Image sobel = sobelFilter(blurF);
    savePNG("mandelbrot_16k_sobel.png", sobel, IMG_W, IMG_H);

    std::cout << "[Tarea B] Listo en " << elapsed(t1) << " s\n";

    double total = elapsed(T0);
    std::cout << "\n============================================\n"
              << "  TOTAL: " << total << " s  (" << total/60.0 << " min)\n"
              << "============================================\n"
              << "\nArchivos generados:\n"
              << "  mandelbrot_16k.png        — fractal Mandelbrot\n"
              << "  mandelbrot_16k_blur.png   — desenfoque Gaussiano\n"
              << "  mandelbrot_16k_sobel.png  — detección de bordes Sobel\n";
    return 0;
}