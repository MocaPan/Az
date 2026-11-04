#include <Arduino.h>
#include <math.h>

// =====================================================
// CONFIGURACIÓN GENERAL
// =====================================================

// UART con Nodo 1
const int PIN_UART_RX = 18;
const int PIN_UART_TX = 17;

const uint32_t BAUD_UART = 460800;

// Salida PWM hacia filtro RC + PAM8403
const int PIN_PWM = 4;

const uint32_t FRECUENCIA_PWM = 62500;
const uint8_t RESOLUCION_PWM = 8;

const uint16_t N = 1024;
const float FS = 16000.0;


// =====================================================
// AJUSTE DE NIVEL DE SALIDA
// =====================================================
//
// Convierte las muestras filtradas a duty PWM.
//
// Si se escucha muy bajo:
// subir poco a poco: 0.12 -> 0.15 -> 0.18
//
// Si distorsiona:
// bajar: 0.12 -> 0.10 -> 0.08
//

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
// FILTRO TEMPORAL IIR
//
// Butterworth
// orden = 12
// fc = 1500 Hz
// fs = 16000 Hz
//
// 6 secciones SOS
// =====================================================

const int NUM_SOS = 6;

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


// =====================================================
// ESTADOS DEL IIR
// =====================================================

float z1[NUM_SOS] = {0};
float z2[NUM_SOS] = {0};


// =====================================================
// BUFFERS DSP
// =====================================================

float entradaTemporal[N];
float salidaTemporal[N];


// =====================================================
// DOBLE BUFFER PARA REPRODUCCIÓN PWM
// =====================================================
//
// estado:
//
// 0 = libre
// 1 = llenando
// 2 = listo
// 3 = reproduciendo
//

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
// BUSCAR CABECERA AA 55
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
// IIR - UNA MUESTRA
//
// Direct Form II Transposed:
//
// y  = b0*x + z1
// z1 = b1*x - a1*y + z2
// z2 = b2*x - a2*y
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


// =====================================================
// FILTRAR BLOQUE
// =====================================================

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

  return sqrt(
    energia / N
  );
}


// =====================================================
// GOERTZEL
//
// Mide amplitud en una sola frecuencia.
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


  if (potencia < 0.0f) {
    potencia = 0.0f;
  }


  return
    (2.0f / N)
    * sqrtf(potencia);
}


// =====================================================
// PREPARAR BLOQUE PWM
// =====================================================

void prepararBloquePWM() {

  int indiceBuffer = -1;


  // Buscar un buffer libre
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


  // No hay buffer libre:
  // no bloqueamos UART.
  if (indiceBuffer < 0) {

    bloquesAudioDescartados++;

    return;
  }


  // Convertir salida IIR a duty PWM

  for (int i = 0; i < N; i++) {

    float valor =
      salidaTemporal[i]
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


  // Marcar como listo
  portENTER_CRITICAL(
    &muxBuffer
  );

  estadoBuffer[indiceBuffer] = 2;

  portEXIT_CRITICAL(
    &muxBuffer
  );
}


// =====================================================
// TAREA DE REPRODUCCIÓN
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


    // Si no hay audio listo:
    if (indiceBuffer < 0) {

      ledcWrite(
        PIN_PWM,
        PWM_CENTRO
      );

      vTaskDelay(
        1
      );

      continue;
    }


    // =====================================
    // Reproducir 1024 muestras @ 16 kHz
    // =====================================

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


      if (sumar63) {

        siguiente += 63;

      } else {

        siguiente += 62;
      }

      sumar63 =
        !sumar63;
    }


    // Bloque consumido
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
// MOSTRAR MÉTRICAS
// =====================================================

void mostrarResultado(
  const PaqueteAudio& p
) {

  float rmsEntrada =
    calcularRMS(
      entradaTemporal
    );

  float rmsSalida =
    calcularRMS(
      salidaTemporal
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
    " | RMS IIR="
  );

  Serial.print(
    rmsSalida,
    1
  );


  // =====================================
  // Métrica tonal
  // =====================================

  if (
    p.valido
    && p.tipoRuido == 1
  ) {

    float ampEntrada =
      amplitudFrecuencia(
        entradaTemporal,
        p.frecuenciaHz,
        p.fs
      );


    float ampSalida =
      amplitudFrecuencia(
        salidaTemporal,
        p.frecuenciaHz,
        p.fs
      );


    float atenuacionTono =
      20.0f
      * log10f(
          ampEntrada
          / max(
              ampSalida,
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
      " | Aten="
    );

    Serial.print(
      atenuacionTono,
      1
    );

    Serial.print(
      " dB"
    );
  }


  Serial.print(
    " | Perdidos="
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
    " | AudioDrop="
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

  // Debug USB
  Serial.begin(
    115200
  );


  // =====================================
  // UART Nodo 1
  // =====================================

  Serial1.setRxBufferSize(
    8192
  );

  Serial1.begin(
    BAUD_UART,
    SERIAL_8N1,
    PIN_UART_RX,
    PIN_UART_TX
  );


  // =====================================
  // PWM
  // =====================================

  ledcAttach(
    PIN_PWM,
    FRECUENCIA_PWM,
    RESOLUCION_PWM
  );

  ledcWrite(
    PIN_PWM,
    PWM_CENTRO
  );


  // =====================================
  // Crear tarea de audio
  // =====================================

  xTaskCreatePinnedToCore(
    tareaAudio,          // función
    "AudioPWM",          // nombre
    4096,                // stack
    NULL,                // parámetro
    2,                   // prioridad
    NULL,                // handle
    0                    // Core 0
  );


  Serial.println();

  Serial.println(
    "================================"
  );

  Serial.println(
    "NODO 2 - IIR + SALIDA PWM"
  );

  Serial.println(
    "================================"
  );

  Serial.println(
    "fs = 16000 Hz"
  );

  Serial.println(
    "Butterworth orden 12"
  );

  Serial.println(
    "fc = 1500 Hz"
  );

  Serial.println(
    "PWM GPIO4"
  );

  Serial.println();
}


// =====================================================
// LOOP PRINCIPAL
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
    // IIR
    // =====================================

    filtrarBloqueTemporal(
      paquete
    );


    // =====================================
    // Preparar salida física
    // =====================================

    prepararBloquePWM();


    // =====================================
    // Debug cada 10 bloques
    // =====================================

    if (
      paquete.numeroBloque
      % 10 == 0
    ) {

      mostrarResultado(
        paquete
      );
    }
  }
}