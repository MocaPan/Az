#include <Arduino.h>
#include <arduinoFFT.h>
#include <math.h>

// =====================================================
// CONFIGURACIÓN
// =====================================================

const int PIN_UART_RX = 18;
const int PIN_UART_TX = 17;

const uint32_t BAUD_UART = 460800;

const uint16_t N = 1024;
const uint16_t HOP = 512;

const float FS = 16000.0f;
const float DELTA_F = FS / N;

// Parámetros heredados de la estrategia espectral
const float REDUCCION = 1.0f;
const float GANANCIA_MIN = 0.08f;
const float PASO_BANDA_HZ = 1450.0f;
const float UMBRAL_PICO = 7.0f;


// =====================================================
// PAQUETE DEL NODO 1
// =====================================================

struct __attribute__((packed)) PaqueteAudio {

  uint8_t magic1;
  uint8_t magic2;

  uint32_t numeroBloque;

  uint16_t fs;
  uint16_t n;

  uint8_t tipoRuido;
  uint8_t valido;

  float frecuenciaHz;
  float magnitud;

  int16_t muestras[N];

  uint16_t crc;
};


// =====================================================
// ESTADÍSTICAS UART
// =====================================================

uint32_t ultimoBloque = 0;
uint32_t bloquesPerdidos = 0;
uint32_t erroresCRC = 0;


// =====================================================
// BUFFERS FFT
// =====================================================

float vReal[N];
float vImag[N];

float magnitud[N / 2 + 1];
float piso[N / 2 + 1];

float ventana[N];

// Perfil espectral adaptativo.
// Sustituye la mediana temporal offline del Python
// por una estimación progresiva en tiempo real.
float perfil[N / 2 + 1];

bool perfilInicializado = false;


// =====================================================
// BUFFERS DE STREAMING
// =====================================================

// Últimas 512 muestras del bloque anterior.
float mitadAnterior[HOP] = {0};

// Salida del método espectral para el bloque actual.
float salidaEspectral[N];

// Cola de overlap-add.
float colaOLA[HOP] = {0};

bool primeraTrama = true;


// =====================================================
// FFT
// =====================================================

ArduinoFFT<float> FFT(
  vReal,
  vImag,
  N,
  FS
);


// =====================================================
// CRC16
// =====================================================

uint16_t calcularCRC16(
  const uint8_t* datos,
  size_t longitud
) {

  uint16_t crc = 0xFFFF;

  for (size_t i = 0; i < longitud; i++) {

    crc ^= ((uint16_t)datos[i] << 8);

    for (uint8_t bit = 0; bit < 8; bit++) {

      if (crc & 0x8000) {
        crc = (crc << 1) ^ 0x1021;
      } else {
        crc <<= 1;
      }
    }
  }

  return crc;
}


// =====================================================
// BUSCAR CABECERA
// =====================================================

bool buscarInicioPaquete() {

  static uint8_t anterior = 0;

  while (Serial1.available()) {

    uint8_t actual =
      Serial1.read();

    if (
      anterior == 0xAA
      && actual == 0x55
    ) {

      return true;
    }

    anterior = actual;
  }

  return false;
}


// =====================================================
// RECIBIR PAQUETE
// =====================================================

bool recibirPaquete(
  PaqueteAudio& paquete
) {

  if (!buscarInicioPaquete()) {
    return false;
  }

  paquete.magic1 = 0xAA;
  paquete.magic2 = 0x55;


  uint8_t* destino =
    ((uint8_t*)&paquete) + 2;

  size_t faltan =
    sizeof(PaqueteAudio) - 2;

  size_t recibidos = 0;

  unsigned long inicio =
    millis();


  while (recibidos < faltan) {

    if (Serial1.available()) {

      destino[recibidos++] =
        Serial1.read();

    } else if (
      millis() - inicio > 100
    ) {

      return false;
    }
  }

  return true;
}


// =====================================================
// MEDIANA DE VENTANA PEQUEÑA
// =====================================================

float medianaLocal(
  int centro
) {

  float valores[31];

  int cantidad = 0;


  for (int d = -15; d <= 15; d++) {

    int k =
      centro + d;

    if (k < 0) {
      k = 0;
    }

    if (k > N / 2) {
      k = N / 2;
    }

    valores[cantidad++] =
      magnitud[k];
  }


  // Ordenamiento simple.
  // Solo son 31 elementos.
  for (int i = 1; i < cantidad; i++) {

    float clave =
      valores[i];

    int j =
      i - 1;

    while (
      j >= 0
      && valores[j] > clave
    ) {

      valores[j + 1] =
        valores[j];

      j--;
    }

    valores[j + 1] =
      clave;
  }


  return
    valores[cantidad / 2];
}


// =====================================================
// MEDIANA DEL PERFIL
// =====================================================

float medianaPerfil(
  int centro
) {

  float valores[31];

  int cantidad = 0;


  for (int d = -15; d <= 15; d++) {

    int k =
      centro + d;

    if (k < 0) {
      k = 0;
    }

    if (k > N / 2) {
      k = N / 2;
    }

    valores[cantidad++] =
      perfil[k];
  }


  for (int i = 1; i < cantidad; i++) {

    float clave =
      valores[i];

    int j =
      i - 1;

    while (
      j >= 0
      && valores[j] > clave
    ) {

      valores[j + 1] =
        valores[j];

      j--;
    }

    valores[j + 1] =
      clave;
  }


  return
    valores[cantidad / 2];
}


// =====================================================
// PROCESAR UNA TRAMA FFT/IFFT
//
// Devuelve HOP = 512 muestras nuevas.
// =====================================================

void procesarTrama(
  const float* entrada,
  float* salidaHop
) {

  // =========================================
  // 1. sqrt-Hann
  // =========================================

  for (int i = 0; i < N; i++) {

    vReal[i] =
      entrada[i]
      * ventana[i];

    vImag[i] =
      0.0f;
  }


  // =========================================
  // 2. FFT
  // =========================================

  FFT.compute(
    FFTDirection::Forward
  );


  // =========================================
  // 3. Magnitud unilateral
  // =========================================

  float maxMag = 0.0f;

  for (int k = 0; k <= N / 2; k++) {

    magnitud[k] =
      sqrtf(
        vReal[k] * vReal[k]
        + vImag[k] * vImag[k]
      );

    if (magnitud[k] > maxMag) {
      maxMag = magnitud[k];
    }
  }


  // =========================================
  // 4. Piso espectral local
  // =========================================

  for (int k = 0; k <= N / 2; k++) {

    piso[k] =
      medianaLocal(k);
  }


  // =========================================
  // 5. Perfil adaptativo
  // =========================================

  if (!perfilInicializado) {

    for (int k = 0; k <= N / 2; k++) {

      perfil[k] =
        magnitud[k];
    }

    perfilInicializado =
      true;

  } else {

    const float ALPHA = 0.90f;

    for (int k = 0; k <= N / 2; k++) {

      perfil[k] =
        ALPHA * perfil[k]
        + (1.0f - ALPHA)
          * magnitud[k];
    }
  }


  // =========================================
  // 6. Ganancia espectral
  // =========================================

  for (int k = 0; k <= N / 2; k++) {

    float mag =
      magnitud[k];

    float gain =
      1.0f
      - REDUCCION
        * (
            piso[k]
            / (mag + 1e-12f)
          );


    if (gain < GANANCIA_MIN) {

      gain =
        GANANCIA_MIN;
    }


    // ---------------------------------------
    // Detección automática de pico
    // ---------------------------------------

    float frecuencia =
      k * DELTA_F;


    bool bandaElegible =
      frecuencia > PASO_BANDA_HZ
      && frecuencia
         < 0.95f * (FS / 2.0f);


    bool suficientementeGrande =
      mag
      > 0.01f * maxMag;


    float fondoPerfil =
      medianaPerfil(k);


    float razon =
      perfil[k]
      / (fondoPerfil + 1e-12f);


    bool picoLocal = false;


    if (
      k > 0
      && k < N / 2
    ) {

      picoLocal =
        perfil[k]
          > perfil[k - 1]
        &&
        perfil[k]
          > perfil[k + 1];
    }


    if (
      bandaElegible
      && suficientementeGrande
      && picoLocal
      && razon >= UMBRAL_PICO
    ) {

      gain *=
        1.0f
        / (
            1.0f
            + 8.0f * REDUCCION
          );
    }


    // ---------------------------------------
    // Aplicar a k
    // ---------------------------------------

    vReal[k] *= gain;
    vImag[k] *= gain;


    // ---------------------------------------
    // Preservar simetría conjugada
    // ---------------------------------------

    if (
      k > 0
      && k < N / 2
    ) {

      int espejo =
        N - k;

      vReal[espejo] *= gain;
      vImag[espejo] *= gain;
    }
  }


  // =========================================
  // 7. IFFT
  // =========================================

  FFT.compute(
    FFTDirection::Reverse
  );


  // ArduinoFFT normaliza automáticamente
  // la transformada inversa.


  // =========================================
  // 8. Ventana de síntesis sqrt-Hann
  // =========================================

  for (int i = 0; i < N; i++) {

    vReal[i] *=
      ventana[i];
  }


  // =========================================
  // 9. Overlap-add
  // =========================================

  for (int i = 0; i < HOP; i++) {

    salidaHop[i] =
      vReal[i]
      + colaOLA[i];

    colaOLA[i] =
      vReal[i + HOP];
  }
}


// =====================================================
// PROCESAR BLOQUE COMPLETO
//
// Cada paquete de 1024 muestras produce
// dos tramas con salto de 512.
// =====================================================

void procesarBloqueEspectral(
  const PaqueteAudio& p
) {

  float trama[N];

  float hopSalida[HOP];


  // =========================================
  // TRAMA 1
  //
  // mitad anterior + primera mitad actual
  // =========================================

  for (int i = 0; i < HOP; i++) {

    trama[i] =
      mitadAnterior[i];

    trama[i + HOP] =
      (float)p.muestras[i];
  }


  procesarTrama(
    trama,
    hopSalida
  );


  for (int i = 0; i < HOP; i++) {

    salidaEspectral[i] =
      hopSalida[i];
  }


  // =========================================
  // TRAMA 2
  //
  // bloque actual completo
  // =========================================

  for (int i = 0; i < N; i++) {

    trama[i] =
      (float)p.muestras[i];
  }


  procesarTrama(
    trama,
    hopSalida
  );


  for (int i = 0; i < HOP; i++) {

    salidaEspectral[i + HOP] =
      hopSalida[i];
  }


  // =========================================
  // Guardar segunda mitad para próximo bloque
  // =========================================

  for (int i = 0; i < HOP; i++) {

    mitadAnterior[i] =
      (float)p.muestras[i + HOP];
  }
}


// =====================================================
// RMS
// =====================================================

float calcularRMSInt16(
  const int16_t* señal
) {

  double energia = 0.0;

  for (int i = 0; i < N; i++) {

    double x =
      señal[i];

    energia +=
      x * x;
  }


  return
    sqrt(
      energia / N
    );
}


float calcularRMSFloat(
  const float* señal
) {

  double energia = 0.0;

  for (int i = 0; i < N; i++) {

    double x =
      señal[i];

    energia +=
      x * x;
  }


  return
    sqrt(
      energia / N
    );
}


// =====================================================
// GOERTZEL
// =====================================================

float amplitudFrecuenciaInt16(
  const int16_t* senal,
  float frecuencia,
  float fs
) {

  float omega =
    2.0f * PI
    * frecuencia / fs;

  float coef =
    2.0f * cosf(omega);

  float s0 = 0.0f;
  float s1 = 0.0f;
  float s2 = 0.0f;


  for (int n = 0; n < N; n++) {

    s0 =
      senal[n]
      + coef * s1
      - s2;

    s2 = s1;
    s1 = s0;
  }


  float potencia =
    s1 * s1
    + s2 * s2
    - coef * s1 * s2;


  if (potencia < 0.0f) {
    potencia = 0.0f;
  }


  return
    (2.0f / N)
    * sqrtf(potencia);
}


float amplitudFrecuenciaFloat(
  const float* senal,
  float frecuencia,
  float fs
) {

  float omega =
    2.0f * PI
    * frecuencia / fs;

  float coef =
    2.0f * cosf(omega);

  float s0 = 0.0f;
  float s1 = 0.0f;
  float s2 = 0.0f;


  for (int n = 0; n < N; n++) {

    s0 =
      senal[n]
      + coef * s1
      - s2;

    s2 = s1;
    s1 = s0;
  }


  float potencia =
    s1 * s1
    + s2 * s2
    - coef * s1 * s2;


  if (potencia < 0.0f) {
    potencia = 0.0f;
  }


  return
    (2.0f / N)
    * sqrtf(potencia);
}


// =====================================================
// MOSTRAR RESULTADOS
// =====================================================

void mostrarResultado(
  const PaqueteAudio& p,
  unsigned long tiempoUs
) {

  float rmsEntrada =
    calcularRMSInt16(
      p.muestras
    );


  float rmsEspectral =
    calcularRMSFloat(
      salidaEspectral
    );


  Serial.print(
    "Bloque "
  );

  Serial.print(
    p.numeroBloque
  );


  Serial.print(
    " | RMS in="
  );

  Serial.print(
    rmsEntrada,
    1
  );


  Serial.print(
    " | RMS ESP="
  );

  Serial.print(
    rmsEspectral,
    1
  );


  if (
    p.valido
    && p.tipoRuido == 1
  ) {

    float Ain =
      amplitudFrecuenciaInt16(
        p.muestras,
        p.frecuenciaHz,
        p.fs
      );


    float Aout =
      amplitudFrecuenciaFloat(
        salidaEspectral,
        p.frecuenciaHz,
        p.fs
      );


    float atenuacion =
      20.0f
      * log10f(
          Ain
          / max(
              Aout,
              0.000001f
            )
        );


    Serial.print(
      " | TONAL "
    );

    Serial.print(
      p.frecuenciaHz,
      1
    );

    Serial.print(
      " Hz"
    );


    Serial.print(
      " | AtenESP="
    );

    Serial.print(
      atenuacion,
      1
    );

    Serial.print(
      " dB"
    );
  }


  Serial.print(
    " | DSP="
  );

  Serial.print(
    tiempoUs / 1000.0f,
    2
  );

  Serial.print(
    " ms"
  );


  Serial.print(
    " | Perdidos="
  );

  Serial.print(
    bloquesPerdidos
  );


  Serial.print(
    " | CRC="
  );

  Serial.println(
    erroresCRC
  );
}


// =====================================================
// SETUP
// =====================================================

void setup() {

  Serial.begin(115200);


  Serial1.setRxBufferSize(
    8192
  );

  Serial1.begin(
    BAUD_UART,
    SERIAL_8N1,
    PIN_UART_RX,
    PIN_UART_TX
  );


  // =========================================
  // sqrt-Hann
  // =========================================

  for (int i = 0; i < N; i++) {

    float hann =
      0.5f
      * (
          1.0f
          - cosf(
              2.0f
              * PI
              * i
              / N
            )
        );

    ventana[i] =
      sqrtf(
        max(
          hann,
          0.0f
        )
      );
  }


  Serial.println();

  Serial.println(
    "================================"
  );

  Serial.println(
    "NODO 3 - FILTRADO ESPECTRAL"
  );

  Serial.println(
    "================================"
  );

  Serial.println(
    "FFT/IFFT N=1024"
  );

  Serial.println(
    "Overlap = 50%"
  );

  Serial.println(
    "sqrt-Hann + OLA"
  );

  Serial.println(
    "Reduccion = 1.0"
  );

  Serial.println();
}


// =====================================================
// LOOP
// =====================================================

void loop() {

  static PaqueteAudio paquete;


  if (
    recibirPaquete(
      paquete
    )
  ) {

    // =====================================
    // CRC
    // =====================================

    uint16_t crcCalculado =
      calcularCRC16(
        (uint8_t*)&paquete,
        sizeof(PaqueteAudio)
        - sizeof(uint16_t)
      );


    if (
      crcCalculado
      != paquete.crc
    ) {

      erroresCRC++;

      return;
    }


    // =====================================
    // Pérdida de bloques
    // =====================================

    if (
      ultimoBloque != 0
      && paquete.numeroBloque
         > ultimoBloque + 1
    ) {

      bloquesPerdidos +=
        paquete.numeroBloque
        - ultimoBloque
        - 1;
    }


    ultimoBloque =
      paquete.numeroBloque;


    // =====================================
    // FFT / IFFT
    // =====================================

    unsigned long inicioDSP =
      micros();


    procesarBloqueEspectral(
      paquete
    );


    unsigned long tiempoDSP =
      micros()
      - inicioDSP;


    // =====================================
    // Debug cada 10 bloques
    // =====================================

    if (
      paquete.numeroBloque
      % 10 == 0
    ) {

      mostrarResultado(
        paquete,
        tiempoDSP
      );
    }
  }
}