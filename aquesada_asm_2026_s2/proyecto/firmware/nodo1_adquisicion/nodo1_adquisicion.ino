#include <Arduino.h>
#include <arduinoFFT.h>

// =====================================================
// CONFIGURACIÓN GENERAL
// =====================================================

const int PIN_MIC = 34;

// UART hacia Nodo 2/3
const int PIN_UART_RX = 16;
const int PIN_UART_TX = 17;

const uint32_t BAUD_UART = 460800;

const uint16_t N = 1024;
const float FS = 16000.0;
const float DELTA_F = FS / N;


// =====================================================
// FFT
// =====================================================

float vReal[N];
float vImag[N];
int16_t bloqueAudio[N];

ArduinoFFT<float> FFT(
  vReal,
  vImag,
  N,
  FS
);


// =====================================================
// DIAGNÓSTICO
// =====================================================

const float UMBRAL_MAGNITUD = 8000.0;
const float TOLERANCIA_HZ = 25.0;
const int BLOQUES_CONFIRMACION = 3;

const float FREQ_MIN = 100.0;
const float FREQ_MAX = 7000.0;

float frecuenciaAnterior = 0.0;
int contadorConfirmacion = 0;


// =====================================================
// ESTADO
// =====================================================

uint32_t numeroBloque = 0;

float frecuenciaDetectada = 0.0;
float magnitudDetectada = 0.0;

bool interferenciaValida = false;


// =====================================================
// TIPOS DE RUIDO
// =====================================================

enum TipoRuido : uint8_t {
  RUIDO_NO_DETECTADO = 0,
  RUIDO_TONAL = 1
};

TipoRuido tipoRuidoActual =
  RUIDO_NO_DETECTADO;


// =====================================================
// PAQUETE UART
// =====================================================

struct __attribute__((packed)) PaqueteAudio {

  // Delimitación
  uint8_t magic1;
  uint8_t magic2;

  // Identificación
  uint32_t numeroBloque;

  // Configuración
  uint16_t fs;
  uint16_t n;

  // Diagnóstico
  uint8_t tipoRuido;
  uint8_t valido;

  float frecuenciaHz;
  float magnitud;

  // Audio centrado en cero
  int16_t muestras[N];

  // Integridad
  uint16_t crc;
};


// =====================================================
// CRC16-CCITT
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
// CAPTURA ADC A ~16 kHz
// =====================================================

float capturarBloque() {

  unsigned long inicio = micros();
  unsigned long siguienteMuestra = inicio;

  bool sumar63 = false;

  int32_t suma = 0;

  // =========================================
  // Captura ADC
  // =========================================

  for (int i = 0; i < N; i++) {

    while (
      (int32_t)(micros() - siguienteMuestra) < 0
    ) {
    }

    int16_t muestra =
      analogRead(PIN_MIC);

    bloqueAudio[i] = muestra;

    suma += muestra;

    if (sumar63) {
      siguienteMuestra += 63;
    } else {
      siguienteMuestra += 62;
    }

    sumar63 = !sumar63;
  }


  // =========================================
  // Remover DC y preparar FFT
  // =========================================

  float promedio =
    (float)suma / N;

  for (int i = 0; i < N; i++) {

    int16_t centrada =
      (int16_t)(
        bloqueAudio[i] - promedio
      );

    // Esta es la señal que enviaremos
    bloqueAudio[i] = centrada;

    // Y también la señal usada en FFT
    vReal[i] = centrada;
    vImag[i] = 0.0;
  }


  unsigned long fin = micros();

  float tiempo =
    (fin - inicio)
    / 1000000.0;

  return N / tiempo;
}


// =====================================================
// DIAGNÓSTICO
// =====================================================

void procesarDiagnostico(
  float frecuenciaPico,
  float magnitudPico
) {

  if (magnitudPico < UMBRAL_MAGNITUD) {

    contadorConfirmacion = 0;
    frecuenciaAnterior = 0.0;

    interferenciaValida = false;
    tipoRuidoActual = RUIDO_NO_DETECTADO;

    frecuenciaDetectada = 0.0;
    magnitudDetectada = magnitudPico;

    return;
  }

  if (
    abs(frecuenciaPico - frecuenciaAnterior)
    <= TOLERANCIA_HZ
  ) {

    contadorConfirmacion++;

  } else {

    frecuenciaAnterior = frecuenciaPico;
    contadorConfirmacion = 1;
  }

  frecuenciaDetectada = frecuenciaPico;
  magnitudDetectada = magnitudPico;

  if (
    contadorConfirmacion
    >= BLOQUES_CONFIRMACION
  ) {

    interferenciaValida = true;
    tipoRuidoActual = RUIDO_TONAL;

  } else {

    interferenciaValida = false;
    tipoRuidoActual = RUIDO_NO_DETECTADO;
  }
}


// =====================================================
// ENVIAR PAQUETE
// =====================================================

void enviarPaquete() {

  static PaqueteAudio paquete;

  paquete.magic1 = 0xAA;
  paquete.magic2 = 0x55;

  paquete.numeroBloque =
    numeroBloque;

  paquete.fs =
    (uint16_t)FS;

  paquete.n =
    N;

  paquete.tipoRuido =
    (uint8_t)tipoRuidoActual;

  paquete.valido =
    interferenciaValida ? 1 : 0;

  paquete.frecuenciaHz =
    frecuenciaDetectada;

  paquete.magnitud =
    magnitudDetectada;


  // Copiar las 1024 muestras
  for (int i = 0; i < N; i++) {

    paquete.muestras[i] =
      bloqueAudio[i];
  }


  paquete.crc =
    calcularCRC16(
      (uint8_t*)&paquete,
      sizeof(PaqueteAudio)
      - sizeof(uint16_t)
    );


  Serial2.write(
    (uint8_t*)&paquete,
    sizeof(PaqueteAudio)
  );
}


// =====================================================
// DEBUG LOCAL
// =====================================================

void mostrarDiagnostico(
  float fsReal
) {

  Serial.print("Bloque ");
  Serial.print(numeroBloque);

  Serial.print(" | fs = ");
  Serial.print(fsReal, 1);

  Serial.print(" Hz | ");

  if (interferenciaValida) {

    Serial.print("TONAL CONFIRMADA");

    Serial.print(" | f = ");
    Serial.print(
      frecuenciaDetectada,
      1
    );

    Serial.print(" Hz");

    Serial.print(" | mag = ");
    Serial.println(
      magnitudDetectada,
      1
    );

  } else if (
    contadorConfirmacion > 0
  ) {

    Serial.print("Candidato");

    Serial.print(" | f = ");
    Serial.print(
      frecuenciaDetectada,
      1
    );

    Serial.print(
      " Hz | confirmacion "
    );

    Serial.print(
      contadorConfirmacion
    );

    Serial.print("/");
    Serial.println(
      BLOQUES_CONFIRMACION
    );

  } else {

    Serial.print(
      "Sin componente dominante"
    );

    Serial.print(
      " | mag max = "
    );

    Serial.println(
      magnitudDetectada,
      1
    );
  }
}


// =====================================================
// SETUP
// =====================================================

void setup() {

  Serial.begin(115200);

  Serial2.setTxBufferSize(4096);

  // UART2 del ESP32 clásico
  Serial2.begin(
    BAUD_UART,
    SERIAL_8N1,
    PIN_UART_RX,
    PIN_UART_TX
  );

  analogReadResolution(12);

  Serial.println();
  Serial.println(
    "================================"
  );

  Serial.println(
    "NODO 1 - ADQUISICION Y ANALISIS"
  );

  Serial.println(
    "================================"
  );

  Serial.print("fs = ");
  Serial.println(FS);

  Serial.print("N = ");
  Serial.println(N);

  Serial.print("Delta f = ");
  Serial.print(DELTA_F, 3);
  Serial.println(" Hz");

  Serial.print(
    "Banda analizada = "
  );

  Serial.print(FREQ_MIN);
  Serial.print(" - ");
  Serial.print(FREQ_MAX);
  Serial.println(" Hz");

  Serial.println();
}


// =====================================================
// LOOP
// =====================================================

void loop() {

  numeroBloque++;

  // 1. Captura
  float fsReal =
    capturarBloque();

  // 2. Eliminar DC
  //FFT.dcRemoval();

  // 3. Hann
  FFT.windowing(
    FFTWindow::Hann,
    FFTDirection::Forward
  );

  // 4. FFT
  FFT.compute(
    FFTDirection::Forward
  );

  // 5. Magnitud
  FFT.complexToMagnitude();

  vReal[0] = 0;


  // 6. Pico dentro de banda válida

  float frecuenciaPico = 0.0;
  float magnitudPico = 0.0;

  int binMin =
    (int)(FREQ_MIN / DELTA_F);

  int binMax =
    (int)(FREQ_MAX / DELTA_F);

  if (binMin < 1) {
    binMin = 1;
  }

  if (
    binMax > (N / 2 - 1)
  ) {
    binMax = N / 2 - 1;
  }

  for (
    int k = binMin;
    k <= binMax;
    k++
  ) {

    if (
      vReal[k]
      > magnitudPico
    ) {

      magnitudPico =
        vReal[k];

      frecuenciaPico =
        k * DELTA_F;
    }
  }


  // 7. Diagnóstico
  procesarDiagnostico(
    frecuenciaPico,
    magnitudPico
  );


  // 8. Debug
  mostrarDiagnostico(
    fsReal
  );


  // 9. UART
  enviarPaquete();
}