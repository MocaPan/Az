#include <Arduino.h>
#include <arduinoFFT.h>
#include <math.h>

// =====================================================
// CONFIGURACIÓN GENERAL
// =====================================================

// UART desde Nodo 1
const int PIN_UART_RX = 18;
const int PIN_UART_TX = 17;

const uint32_t BAUD_UART = 460800;

// Salida física
const int PIN_PWM = 4;

const uint32_t FRECUENCIA_PWM = 62500;
const uint8_t RESOLUCION_PWM = 8;

// Procesamiento
const uint16_t N = 1024;
const uint16_t HOP = 512;

const float FS = 16000.0f;
const float DELTA_F = FS / N;


// =====================================================
// SELECCIÓN DE MÉTODO
//
// T = temporal
// E = espectral
// =====================================================

volatile char modoSalida = 'T';


// =====================================================
// PWM
// =====================================================

const float GANANCIA_PWM = 0.12f;

const int PWM_CENTRO = 128;
const int PWM_MAX_DESVIO = 110;


// =====================================================
// PAQUETE RECIBIDO
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
// =====================================================
// RUTA 1: FILTRO TEMPORAL IIR
// =====================================================
// =====================================================

const int NUM_SOS = 6;

// Butterworth orden 12
// fc = 1500 Hz
// fs = 16000 Hz
//
// Formato:
// b0, b1, b2, a1, a2

const float sos[NUM_SOS][5] = {

  {
    6.08370757e-08,
    1.21674151e-07,
    6.08370757e-08,
    -1.07229864,
    0.28964260
  },

  {
    1.0,
    2.0,
    1.0,
    -1.09889727,
    0.32163251
  },

  {
    1.0,
    2.0,
    1.0,
    -1.15420694,
    0.38815288
  },

  {
    1.0,
    2.0,
    1.0,
    -1.24265964,
    0.49453405
  },

  {
    1.0,
    2.0,
    1.0,
    -1.37137466,
    0.64933827
  },

  {
    1.0,
    2.0,
    1.0,
    -1.55050228,
    0.86477323
  }
};


// Estados del IIR.
// Se conservan entre bloques.

float z1[NUM_SOS] = {0};
float z2[NUM_SOS] = {0};


// Buffers temporales

float entradaTemporal[N];
float salidaTemporal[N];


// =====================================================
// =====================================================
// RUTA 2: FILTRADO ESPECTRAL
// =====================================================
// =====================================================

const float REDUCCION = 1.0f;
const float GANANCIA_MIN = 0.08f;

const float PASO_BANDA_HZ = 1450.0f;
const float UMBRAL_PICO = 7.0f;


// Buffers FFT

float vReal[N];
float vImag[N];

float magnitud[N / 2 + 1];
float piso[N / 2 + 1];

float perfil[N / 2 + 1];

float ventana[N];

bool perfilInicializado = false;


// Streaming / overlap-add

float mitadAnterior[HOP] = {0};
float colaOLA[HOP] = {0};

float salidaEspectral[N];


// Buffers auxiliares globales.
// Así evitamos poner ~6 kB en el stack.

float tramaEspectral[N];
float hopEspectral[HOP];


// FFT

ArduinoFFT<float> FFT(
  vReal,
  vImag,
  N,
  FS
);


// =====================================================
// DOBLE BUFFER PARA PWM
//
// 0 = libre
// 1 = llenando
// 2 = listo
// 3 = reproduciendo
// =====================================================

uint8_t bufferPWM[2][N];

volatile uint8_t estadoBuffer[2] = {
  0,
  0
};

portMUX_TYPE muxBuffer =
  portMUX_INITIALIZER_UNLOCKED;

uint32_t bloquesAudioDescartados = 0;
uint32_t muestrasClipping = 0;


// =====================================================
// CRC16-CCITT
// =====================================================

uint16_t calcularCRC16(
  const uint8_t* datos,
  size_t longitud
) {

  uint16_t crc = 0xFFFF;

  for (size_t i = 0; i < longitud; i++) {

    crc ^=
      ((uint16_t)datos[i] << 8);

    for (uint8_t bit = 0; bit < 8; bit++) {

      if (crc & 0x8000) {

        crc =
          (crc << 1)
          ^ 0x1021;

      } else {

        crc <<= 1;
      }
    }
  }

  return crc;
}


// =====================================================
// BUSCAR CABECERA UART
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

  const size_t faltan =
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

      Serial.println(
        "ERROR: timeout UART"
      );

      return false;
    }
  }

  return true;
}


// =====================================================
// =====================================================
// FILTRO TEMPORAL
// =====================================================
// =====================================================

float filtrarMuestra(
  float x
) {

  float y = x;

  for (int s = 0; s < NUM_SOS; s++) {

    float entrada = y;

    float b0 = sos[s][0];
    float b1 = sos[s][1];
    float b2 = sos[s][2];

    float a1 = sos[s][3];
    float a2 = sos[s][4];


    y =
      b0 * entrada
      + z1[s];


    z1[s] =
      b1 * entrada
      - a1 * y
      + z2[s];


    z2[s] =
      b2 * entrada
      - a2 * y;
  }

  return y;
}


void filtrarBloqueTemporal(
  const PaqueteAudio& p
) {

  for (int i = 0; i < N; i++) {

    entradaTemporal[i] =
      (float)p.muestras[i];

    salidaTemporal[i] =
      filtrarMuestra(
        entradaTemporal[i]
      );
  }
}


// =====================================================
// =====================================================
// FILTRADO ESPECTRAL
// =====================================================
// =====================================================


// =====================================================
// MEDIANA LOCAL DE MAGNITUD
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
// =====================================================

void procesarTrama(
  const float* entrada,
  float* salidaHop
) {

  // -----------------------------------------
  // 1. sqrt-Hann análisis
  // -----------------------------------------

  for (int i = 0; i < N; i++) {

    vReal[i] =
      entrada[i]
      * ventana[i];

    vImag[i] =
      0.0f;
  }


  // -----------------------------------------
  // 2. FFT
  // -----------------------------------------

  FFT.compute(
    FFTDirection::Forward
  );


  // -----------------------------------------
  // 3. Magnitud
  // -----------------------------------------

  float maxMag = 0.0f;


  for (int k = 0; k <= N / 2; k++) {

    magnitud[k] =
      sqrtf(
        vReal[k] * vReal[k]
        + vImag[k] * vImag[k]
      );


    if (
      magnitud[k] > maxMag
    ) {

      maxMag =
        magnitud[k];
    }
  }


  // -----------------------------------------
  // 4. Piso espectral
  // -----------------------------------------

  for (int k = 0; k <= N / 2; k++) {

    piso[k] =
      medianaLocal(k);
  }


  // -----------------------------------------
  // 5. Perfil adaptativo
  // -----------------------------------------

  if (
    !perfilInicializado
  ) {

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
        ALPHA
        * perfil[k]

        + (1.0f - ALPHA)
        * magnitud[k];
    }
  }


  // -----------------------------------------
  // 6. Ganancia adaptativa
  // -----------------------------------------

  for (int k = 0; k <= N / 2; k++) {

    float mag =
      magnitud[k];


    float gain =
      1.0f
      - REDUCCION
      * (
          piso[k]
          / (
              mag
              + 1e-12f
            )
        );


    if (
      gain < GANANCIA_MIN
    ) {

      gain =
        GANANCIA_MIN;
    }


    float frecuencia =
      k * DELTA_F;


    bool bandaElegible =
      frecuencia > PASO_BANDA_HZ
      &&
      frecuencia
        < 0.95f
          * (FS / 2.0f);


    bool suficientementeGrande =
      mag
      > 0.01f
        * maxMag;


    float fondoPerfil =
      medianaPerfil(k);


    float razon =
      perfil[k]
      / (
          fondoPerfil
          + 1e-12f
        );


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
            + 8.0f
            * REDUCCION
          );
    }


    // Aplicar ganancia

    vReal[k] *= gain;
    vImag[k] *= gain;


    // Preservar simetría conjugada

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


  // -----------------------------------------
  // 7. IFFT
  // -----------------------------------------

  FFT.compute(
    FFTDirection::Reverse
  );


  // -----------------------------------------
  // 8. sqrt-Hann síntesis
  // -----------------------------------------

  for (int i = 0; i < N; i++) {

    vReal[i] *=
      ventana[i];
  }


  // -----------------------------------------
  // 9. Overlap-add
  // -----------------------------------------

  for (int i = 0; i < HOP; i++) {

    salidaHop[i] =
      vReal[i]
      + colaOLA[i];


    colaOLA[i] =
      vReal[i + HOP];
  }
}


// =====================================================
// BLOQUE ESPECTRAL COMPLETO
// =====================================================

void procesarBloqueEspectral(
  const PaqueteAudio& p
) {

  // -----------------------------------------
  // Trama 1:
  // mitad anterior + primera mitad actual
  // -----------------------------------------

  for (int i = 0; i < HOP; i++) {

    tramaEspectral[i] =
      mitadAnterior[i];

    tramaEspectral[i + HOP] =
      (float)p.muestras[i];
  }


  procesarTrama(
    tramaEspectral,
    hopEspectral
  );


  for (int i = 0; i < HOP; i++) {

    salidaEspectral[i] =
      hopEspectral[i];
  }


  // -----------------------------------------
  // Trama 2:
  // bloque actual completo
  // -----------------------------------------

  for (int i = 0; i < N; i++) {

    tramaEspectral[i] =
      (float)p.muestras[i];
  }


  procesarTrama(
    tramaEspectral,
    hopEspectral
  );


  for (int i = 0; i < HOP; i++) {

    salidaEspectral[i + HOP] =
      hopEspectral[i];
  }


  // Guardar mitad final para siguiente bloque

  for (int i = 0; i < HOP; i++) {

    mitadAnterior[i] =
      (float)p.muestras[i + HOP];
  }
}


// =====================================================
// RMS
// =====================================================

float calcularRMS(
  const float* senal
) {

  double energia = 0.0;


  for (int i = 0; i < N; i++) {

    energia +=
      (double)senal[i]
      * senal[i];
  }


  return
    sqrt(
      energia / N
    );
}


// =====================================================
// GOERTZEL
// =====================================================

float amplitudFrecuencia(
  const float* senal,
  float frecuencia,
  float fs
) {

  float omega =
    2.0f
    * PI
    * frecuencia
    / fs;


  float coef =
    2.0f
    * cosf(omega);


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


  if (
    potencia < 0.0f
  ) {

    potencia = 0.0f;
  }


  return
    (2.0f / N)
    * sqrtf(potencia);
}


// =====================================================
// PREPARAR AUDIO PWM
// =====================================================

void prepararBloquePWM(
  const float* fuente
) {

  int indiceBuffer = -1;


  // -----------------------------------------
  // Buscar buffer libre
  // -----------------------------------------

  portENTER_CRITICAL(
    &muxBuffer
  );


  for (int i = 0; i < 2; i++) {

    if (
      estadoBuffer[i] == 0
    ) {

      indiceBuffer = i;

      estadoBuffer[i] = 1;

      break;
    }
  }


  portEXIT_CRITICAL(
    &muxBuffer
  );


  if (
    indiceBuffer < 0
  ) {

    bloquesAudioDescartados++;

    return;
  }


  // -----------------------------------------
  // Convertir a PWM
  // -----------------------------------------

  for (int i = 0; i < N; i++) {

    float valor =
      fuente[i]
      * GANANCIA_PWM;


    if (
      valor > PWM_MAX_DESVIO
    ) {

      valor =
        PWM_MAX_DESVIO;

      muestrasClipping++;
    }


    if (
      valor < -PWM_MAX_DESVIO
    ) {

      valor =
        -PWM_MAX_DESVIO;

      muestrasClipping++;
    }


    int duty =
      PWM_CENTRO
      + (int)roundf(valor);


    duty =
      constrain(
        duty,
        0,
        255
      );


    bufferPWM[indiceBuffer][i] =
      (uint8_t)duty;
  }


  // -----------------------------------------
  // Marcar listo
  // -----------------------------------------

  portENTER_CRITICAL(
    &muxBuffer
  );

  estadoBuffer[indiceBuffer] = 2;

  portEXIT_CRITICAL(
    &muxBuffer
  );
}


// =====================================================
// TAREA DE AUDIO
// =====================================================

void tareaAudio(
  void* parametro
) {

  while (true) {

    int indiceBuffer = -1;


    // Buscar bloque listo

    portENTER_CRITICAL(
      &muxBuffer
    );


    for (int i = 0; i < 2; i++) {

      if (
        estadoBuffer[i] == 2
      ) {

        indiceBuffer = i;

        estadoBuffer[i] = 3;

        break;
      }
    }


    portEXIT_CRITICAL(
      &muxBuffer
    );


    // Sin audio listo

    if (
      indiceBuffer < 0
    ) {

      ledcWrite(
        PIN_PWM,
        PWM_CENTRO
      );

      vTaskDelay(1);

      continue;
    }


    // -----------------------------------------
    // Reproducir @ 16 kHz
    // -----------------------------------------

    unsigned long siguiente =
      micros();

    bool sumar63 = false;


    for (int i = 0; i < N; i++) {

      while (
        (int32_t)(
          micros()
          - siguiente
        ) < 0
      ) {
      }


      ledcWrite(
        PIN_PWM,
        bufferPWM[indiceBuffer][i]
      );


      if (
        sumar63
      ) {

        siguiente += 63;

      } else {

        siguiente += 62;
      }


      sumar63 =
        !sumar63;
    }


    // Liberar buffer

    portENTER_CRITICAL(
      &muxBuffer
    );

    estadoBuffer[indiceBuffer] = 0;

    portEXIT_CRITICAL(
      &muxBuffer
    );
  }
}


// =====================================================
// LEER COMANDOS DEL SERIAL MONITOR
// =====================================================

void revisarComandos() {

  while (
    Serial.available()
  ) {

    char comando =
      Serial.read();


    if (
      comando == 'T'
      || comando == 't'
    ) {

      modoSalida = 'T';

      Serial.println();
      Serial.println(
        ">>> SALIDA TEMPORAL IIR"
      );
      Serial.println();
    }


    if (
      comando == 'E'
      || comando == 'e'
    ) {

      modoSalida = 'E';

      Serial.println();
      Serial.println(
        ">>> SALIDA ESPECTRAL FFT/IFFT"
      );
      Serial.println();
    }
  }
}


// =====================================================
// MOSTRAR COMPARACIÓN
// =====================================================

void mostrarResultados(
  const PaqueteAudio& p,
  unsigned long tiempoIIR,
  unsigned long tiempoESP
) {

  float rmsEntrada =
    calcularRMS(
      entradaTemporal
    );


  float rmsIIR =
    calcularRMS(
      salidaTemporal
    );


  float rmsESP =
    calcularRMS(
      salidaEspectral
    );


  Serial.print(
    "Bloque "
  );

  Serial.print(
    p.numeroBloque
  );


  Serial.print(
    " | Modo="
  );

  Serial.print(
    modoSalida == 'T'
      ? "TEMP"
      : "ESP"
  );


  Serial.print(
    " | RMS in="
  );

  Serial.print(
    rmsEntrada,
    1
  );


  Serial.print(
    " | IIR="
  );

  Serial.print(
    rmsIIR,
    1
  );


  Serial.print(
    " | ESP="
  );

  Serial.print(
    rmsESP,
    1
  );


  // =========================================
  // Atenuación tonal
  // =========================================

  if (
    p.valido
    && p.tipoRuido == 1
  ) {

    float Ain =
      amplitudFrecuencia(
        entradaTemporal,
        p.frecuenciaHz,
        p.fs
      );


    float AIIR =
      amplitudFrecuencia(
        salidaTemporal,
        p.frecuenciaHz,
        p.fs
      );


    float AESP =
      amplitudFrecuencia(
        salidaEspectral,
        p.frecuenciaHz,
        p.fs
      );


    float atenIIR =
      20.0f
      * log10f(
          Ain
          / max(
              AIIR,
              0.000001f
            )
        );


    float atenESP =
      20.0f
      * log10f(
          Ain
          / max(
              AESP,
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
      " | AtIIR="
    );

    Serial.print(
      atenIIR,
      1
    );

    Serial.print(
      " dB"
    );


    Serial.print(
      " | AtESP="
    );

    Serial.print(
      atenESP,
      1
    );

    Serial.print(
      " dB"
    );
  }


  // =========================================
  // Tiempos
  // =========================================

  Serial.print(
    " | tIIR="
  );

  Serial.print(
    tiempoIIR / 1000.0f,
    2
  );

  Serial.print(
    "ms"
  );


  Serial.print(
    " | tESP="
  );

  Serial.print(
    tiempoESP / 1000.0f,
    2
  );

  Serial.print(
    "ms"
  );


  // =========================================
  // Comunicación
  // =========================================

  Serial.print(
    " | Perd="
  );

  Serial.print(
    bloquesPerdidos
  );


  Serial.print(
    " | CRC="
  );

  Serial.print(
    erroresCRC
  );


  Serial.print(
    " | Drop="
  );

  Serial.print(
    bloquesAudioDescartados
  );


  Serial.print(
    " | Clip="
  );

  Serial.println(
    muestrasClipping
  );
}


// =====================================================
// SETUP
// =====================================================

void setup() {

  Serial.begin(
    115200
  );


  // =========================================
  // UART
  // =========================================

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
  // PWM
  // =========================================

  ledcAttach(
    PIN_PWM,
    FRECUENCIA_PWM,
    RESOLUCION_PWM
  );

  ledcWrite(
    PIN_PWM,
    PWM_CENTRO
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


  // =========================================
  // Tarea PWM
  // =========================================

  xTaskCreatePinnedToCore(
    tareaAudio,
    "AudioPWM",
    4096,
    NULL,
    2,
    NULL,
    0
  );


  // =========================================
  // Información
  // =========================================

  Serial.println();
  Serial.println(
    "========================================"
  );

  Serial.println(
    "NODO 2 + NODO 3"
  );

  Serial.println(
    "TEMPORAL + ESPECTRAL"
  );

  Serial.println(
    "========================================"
  );

  Serial.println(
    "T = escuchar filtro temporal"
  );

  Serial.println(
    "E = escuchar filtro espectral"
  );

  Serial.println();

  Serial.println(
    "Modo inicial: TEMPORAL"
  );

  Serial.println();
}


// =====================================================
// LOOP
// =====================================================

void loop() {

  static PaqueteAudio paquete;


  // Revisar T / E
  revisarComandos();


  if (
    recibirPaquete(
      paquete
    )
  ) {

    // =========================================
    // CRC
    // =========================================

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


    // =========================================
    // Bloques perdidos
    // =========================================

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


    // =========================================
    // RUTA TEMPORAL
    // =========================================

    unsigned long inicioIIR =
      micros();


    filtrarBloqueTemporal(
      paquete
    );


    unsigned long tiempoIIR =
      micros()
      - inicioIIR;


    // =========================================
    // RUTA ESPECTRAL
    // =========================================

    unsigned long inicioESP =
      micros();


    procesarBloqueEspectral(
      paquete
    );


    unsigned long tiempoESP =
      micros()
      - inicioESP;


    // =========================================
    // SELECCIONAR SALIDA FÍSICA
    // =========================================

    if (
      modoSalida == 'T'
    ) {

      prepararBloquePWM(
        salidaTemporal
      );

    } else {

      prepararBloquePWM(
        salidaEspectral
      );
    }


    // =========================================
    // Mostrar comparación cada 10 bloques
    // =========================================

    if (
      paquete.numeroBloque
      % 10 == 0
    ) {

      mostrarResultados(
        paquete,
        tiempoIIR,
        tiempoESP
      );
    }


    // Revisar otra vez comandos
    revisarComandos();
  }
}