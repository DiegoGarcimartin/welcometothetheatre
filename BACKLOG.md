# El Teatro de los Cuentos — Backlog & Contexto

App de **cuentacuentos para niños (4–9 años)**. El niño elige un cuento clásico y va
**enseñando objetos a la cámara** que **sustituyen elementos del cuento** (la casa del
cerdito, lo que lleva Caperucita en la cesta…). El objeto detectado aparece en la escena
y el narrador lo teje en la historia. 5 objetos por cuento. Final feliz. Telón.

> Pivote desde la versión anterior (teatro de comedia improvisada estilo Chiquito).
> TODO lo de comedia/Chiquito/risas/improvisación se ELIMINA del proyecto.

## Decisiones cerradas (OK del usuario)
- **3 cuentos**: Los tres cerditos · Caperucita Roja · El gato con botas.
- **Objetos SUSTITUYEN** elementos del cuento (no son props decorativos).
- **Tono**: fiel al cuento + chispa de fantasía, dulce, finales felices y seguros (sin sustos).
- **Idioma**: español de España. Voz lenta y clara de cuentacuentos (canal principal; los peques no leen).
- **Modelo**: el COCO más potente/preciso de libreyolo (NO open-vocab). Reconoce ~80 objetos comunes.
- **Ilustración**: emoji grandes animados + **la foto real del objeto del niño incrustada** en la escena.
- **Estilo**: teatro luminoso — cortinas que se abren sobre un libro de cuentos colorido (a mi criterio).
- **Sonido**: sin música por ahora (solo voz + sonido de "magia").
- **Extra**: pantalla final tipo "librito" con el cuento entero + las fotos de los objetos.

## Arquitectura
- Backend `main.py` (FastAPI): detección COCO (modelo potente, 1 vez al arrancar), `/narrate` por
  cuento+escena+objeto, datos de los 3 cuentos, sirve `/` e `index.html`, `/assets`.
- Frontend `index.html` (1 archivo): portada con 3 cuentos, cortinas→escena, captura de objeto,
  foto incrustada, recap final. TTS del navegador (es-ES, lento).
- Sin tests, sin login, sin BBDD. `requirements.txt` mínimo. `./start.sh`.

## Mecánica por cuento (5 huecos de sustitución)
- **Tres cerditos** 🐷🐷🐷🐺: casa cerdito 1 / casa 2 / casa 3 / con qué espantan al lobo / qué cocinan al final.
- **Caperucita** 🔴🐺👵: qué lleva en la cesta / con qué se disfraza el lobo / "¡qué ___ tan grande!" / con qué ayuda el leñador / qué meriendan todos.
- **El gato con botas** 🐱👢👑: qué se pone en las patas / qué regala al rey / qué consigue para impresionar / con qué despista al ogro / qué sirven en el banquete.

## Tareas (en orden; subir a producción tras cada una)
- [x] **T0** Backlog + contexto (este archivo).
- [x] **T1** Limpieza: borrar clips Chiquito + risas.mp3; quitar código de comedia (géneros, objetos graciosos, risas/clips JS, TTS fish/eleven, prompts Chiquito).
- [x] **T2** Modelo más potente: benchmark (DFINE/DEIM/RTDETR vs YOLO9m), elegir el más preciso con velocidad aceptable, cablearlo.
- [x] **T3** Datos de los 3 cuentos en backend + endpoints `/tales`, `/tale/{id}`.
- [x] **T4** `/narrate` reescrito: cuentacuentos fiel + chispa, sustituye el objeto en el hueco, finales seguros.
- [x] **T5** Frontend visual: teatro luminoso, portada 3 cuentos, escena con emoji + fondo, progreso ⭐, subtítulo grande.
- [x] **T6** Frontend flujo: elegir cuento → cortinas → 5 escenas (narra → "enséñame X" → captura → foto a la escena → narra sustitución) → final → telón.
- [x] **T7** Recap "librito" final con el cuento entero + fotos de objetos + botón "otra vez".
- [x] **T8** Robustez/pulido: fallback de detección suave (sin objetos graciosos), error de cámara, ritmo y voz para niños.
- [x] **T9** README actualizado al nuevo proyecto.

## Estado
- 2026-06-02: T0 hecho. Ejecutando en continuo.
- 2026-06-04: T1–T9 COMPLETADO. App de cuentacuentos infantil lista y en producción.
