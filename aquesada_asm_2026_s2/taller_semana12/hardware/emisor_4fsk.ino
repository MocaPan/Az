#include <Arduino.h>
#include <math.h>

// =====================================================
// CONFIGURACIÓN DEL EMISOR 4-FSK
// =====================================================

const int PIN_SALIDA = 4;

// Frecuencia de muestreo utilizada en el modelo Python
const uint32_t FS = 8000;

// PWM utilizado como salida pseudoanalógica
const uint32_t FRECUENCIA_PWM = 62500;
const uint8_t RESOLUCION_PWM = 8;

// Cada símbolo contiene 512 muestras
const uint16_t N_SIMBOLO = 512;

// Duración:
// 512 / 8000 = 0.064 s = 64 ms
const uint32_t DURACION_SIMBOLO_MS = 64;


// =====================================================
// PLAN DE FRECUENCIAS 4-FSK
// =====================================================
//
// Símbolo 0 = 00 = 1000 Hz
// Símbolo 1 = 01 = 1500 Hz
// Símbolo 2 = 10 = 2000 Hz
// Símbolo 3 = 11 = 2500 Hz
//

const uint32_t frecuencias[4] = {
  1000,
  1500,
  2000,
  2500
};


// =====================================================
// MENSAJE DE PRUEBA
// Igual al ejemplo del prototipo de alto nivel
// =====================================================

const uint8_t mensaje[] = {
  0, 1, 2, 3, 3, 2, 1, 0
};

const int CANTIDAD_SIMBOLOS =
  sizeof(mensaje) / sizeof(mensaje[0]);


// =====================================================
// DDS: TABLA DE SENO + ACUMULADOR DE FASE
// =====================================================

const int TAM_TABLA = 64;

uint8_t tablaSeno[TAM_TABLA];

volatile uint32_t fase = 0;
volatile uint32_t incrementoFase = 0;

volatile bool emitiendo = false;


// =====================================================
// TIMER
// =====================================================

hw_timer_t *timer = NULL;


// =====================================================
// CONVERTIR FRECUENCIA A INCREMENTO DE FASE
// =====================================================

uint32_t calcularIncrementoFase(uint32_t frecuencia) {

  // DDS:
  // incremento = frecuencia / FS * 2^32

  return (uint32_t)(
    ((uint64_t)frecuencia << 32) / FS
  );
}


// =====================================================
// INTERRUPCIÓN A 8 kHz
// =====================================================

void ARDUINO_ISR_ATTR onTimer() {

  if (!emitiendo) {

    // Duty medio -> señal aproximadamente DC después del RC
    ledcWrite(PIN_SALIDA, 128);
    return;
  }

  // Avanzar fase
  fase += incrementoFase;

  // Usamos los 6 bits más significativos porque la tabla tiene 64 = 2^6 posiciones
  uint8_t indice = fase >> 26;

  ledcWrite(PIN_SALIDA, tablaSeno[indice]);
}


// =====================================================
// GENERAR TABLA SENO
// =====================================================

void crearTablaSeno() {

  for (int i = 0; i < TAM_TABLA; i++) {

    float angulo =
      2.0f * PI * i / TAM_TABLA;

    float seno = sinf(angulo);

    // Convertir rango [-1,1] a [0,255]
    tablaSeno[i] =
      (uint8_t)(128.0f + 127.0f * seno);
  }
}


// =====================================================
// EMITIR UN SÍMBOLO 4-FSK
// =====================================================

void transmitirSimbolo(uint8_t simbolo) {

  if (simbolo > 3) {
    return;
  }

  uint32_t frecuencia =
    frecuencias[simbolo];

  incrementoFase =
    calcularIncrementoFase(frecuencia);

  // Reiniciar fase al inicio del símbolo
  fase = 0;

  emitiendo = true;

  Serial.print("Simbolo ");
  Serial.print(simbolo);

  Serial.print(" | bits ");

  switch (simbolo) {

    case 0:
      Serial.print("00");
      break;

    case 1:
      Serial.print("01");
      break;

    case 2:
      Serial.print("10");
      break;

    case 3:
      Serial.print("11");
      break;
  }

  Serial.print(" | ");
  Serial.print(frecuencia);
  Serial.println(" Hz");

  delay(DURACION_SIMBOLO_MS);
}


// =====================================================
// SETUP
// =====================================================

void setup() {

  Serial.begin(115200);

  // Crear seno
  crearTablaSeno();

  // Configurar PWM
  ledcAttach(
    PIN_SALIDA,
    FRECUENCIA_PWM,
    RESOLUCION_PWM
  );

  ledcWrite(PIN_SALIDA, 128);


  // -------------------------------------------------
  // TIMER
  //
  // 8 MHz / 1000 ticks = 8000 interrupciones/s
  // -------------------------------------------------

  timer = timerBegin(8000000);

  timerAttachInterrupt(
    timer,
    &onTimer
  );

  timerAlarm(
    timer,
    1000,
    true,
    0
  );


  Serial.println();
  Serial.println("==============================");
  Serial.println("EMISOR 4-FSK ESP32-S3");
  Serial.println("==============================");

  Serial.println("FS = 8000 Hz");
  Serial.println("N = 512 muestras/simbolo");
  Serial.println("Ts = 64 ms");

  Serial.println();

  Serial.println("00 -> 1000 Hz");
  Serial.println("01 -> 1500 Hz");
  Serial.println("10 -> 2000 Hz");
  Serial.println("11 -> 2500 Hz");

  Serial.println();
}


// =====================================================
// LOOP
// =====================================================

void loop() {

  Serial.println("Transmitiendo mensaje:");
  Serial.println("00 01 10 11 11 10 01 00");
  Serial.println();

  for (int i = 0;
       i < CANTIDAD_SIMBOLOS;
       i++) {

    transmitirSimbolo(
      mensaje[i]
    );
  }

  // Detener emisión entre mensajes
  emitiendo = false;

  Serial.println();
  Serial.println("Fin del mensaje");
  Serial.println("------------------------------");

  // Pausa para distinguir cada repetición
  delay(1500);
}