# Aproximación por Mínimos Cuadrados — Discreta y No Lineal

**Proyecto Informático — Análisis Numérico 2026**
Facultad de Ingeniería — Universidad de Mendoza

---

## Autores

| Nombre | Carrera |
|--------|---------|
| Francisco Martin Gallardo | Ingeniería en Informática |
| Santiago Ariel Martinez | Ingeniería en Informática |
| Luciano Panella | Ingeniería en Informática |

**Docente:** Bioing. Emiliano Aparicio — emiliano.aparicio@um.edu.ar
**Entrega:** Antes del 31 de agosto de 2026

---

## Descripción

Herramienta en Python para resolver problemas de **aproximación discreta por mínimos cuadrados**. Dado un conjunto de puntos medidos (xk, yk), el programa encuentra la función que mejor los representa, calcula el error del método y genera un gráfico comparativo.

Se cubren dos variantes:

**Aproximación discreta lineal:** el usuario define la base de funciones y el programa resuelve el sistema de ecuaciones normales.

**Aproximación discreta no lineal:** cuando la función buscada no es combinación lineal, se aplica una transformación algebraica para reducirla al caso lineal. Se incluyen 4 modelos:

| Modelo | Transformación | Base usada |
|--------|---------------|------------|
| `f(x) = a·e^(b·x + c·x²)` | `z = ln(y)` | `{1, x, x²}` |
| `f(x) = a·cos(b·x + π)` | `g = -cos(b·x)` | `{g(x)}` |
| `f(x) = a / (b + x)` | `z = 1/y` | `{1, x}` |
| `f(x) = a·x / (1 + b·e^(c·x))` | `z = x/y` | `{1, e^(c·x)}` |

---

## Requisitos e instalación

```bash
pip install numpy matplotlib
python minimos_cuadrados.py
```

---

## Menú del programa

```
========================================================
  MÍNIMOS CUADRADOS — Análisis Numérico UM 2026
  Francisco Martin Gallardo | Santiago Ariel Martinez
========================================================
  1. Aproximación discreta
  2. Aproximación discreta no lineal
  3. Ver ejemplos
  0. Salir
```

### Opción 1 — Aproximación discreta

El usuario ingresa los puntos (x, y) y las funciones base. El programa arma y resuelve el sistema de ecuaciones normales.

```
¿Cuántos puntos de datos tenés? 5
  x1: 1   y1: 0.046
  x2: 2   y2: 0.059
  ...
¿Cuántas funciones tiene la base? 3
  phi1(x) = 1
  phi2(x) = x
  phi3(x) = 2*x**2-1
```

### Opción 2 — Aproximación discreta no lineal

Submenú con los 4 modelos. El usuario ingresa sus puntos y, según el modelo, puede necesitar un parámetro adicional (b para el coseno, c para la logística).

```
  1. f(x) = a * e^(b*x + c*x²)        [Exp. cuadrática]
  2. f(x) = a * cos(b*x + π)           [Coseno]
  3. f(x) = a / (b + x)                [Hipérbola]
  4. f(x) = a*x / (1 + b*e^(c*x))     [Logística]
```

### Opción 3 — Ejemplos

```
  1. Discreta         — soplete con Chebyshev
  2. No lineal caso 1 — f(x) = a·e^(b·x + c·x²)
  3. No lineal caso 2 — f(x) = a·cos(b·x + π)
  4. No lineal caso 3 — f(x) = a / (b + x)
  5. No lineal caso 4 — f(x) = a·x / (1 + b·e^(c·x))
  6. Ingeniería       — consumo de servidor vs carga CPU
```

---

## Ejemplo de ingeniería en sistemas

Se modela el **consumo eléctrico de un servidor** en función de la carga del procesador. Los datos provienen de un sensor de potencia medido en laboratorio.

| Carga CPU (%) | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Consumo (W) | 45 | 50 | 57 | 66 | 77 | 91 | 107 | 126 | 148 | 174 | 205 |

**Modelo:** `p(x) = 44.04 · e^(0.01327·x + 0.00002·x²)`

**Aplicación:** con el modelo ajustado se predicen consumos para cargas intermedias, lo que permite dimensionar la fuente de alimentación, planificar la refrigeración y calcular el costo energético bajo distintas cargas de trabajo.

---

## Expresiones disponibles para el usuario

Al ingresar funciones por teclado se pueden usar:

| Escribir | Función |
|----------|---------|
| `sin(x)` | seno |
| `cos(x)` | coseno |
| `exp(x)` | exponencial eˣ |
| `log(x)` | logaritmo natural |
| `sqrt(x)` | raíz cuadrada |
| `x**2`, `x**3` | potencias |
| `pi`, `e` | constantes |

---

## Estructura del código

```
minimos_cuadrados.py
│
├── Herramientas base
│   ├── evaluar()               String matemático → valor numérico
│   ├── crear_funcion()         String → función de Python
│   ├── producto_discreto()     Suma f[k]*g[k]
│   └── resolver_sistema()      Muestra y resuelve A·X = B
│
├── Utilidades no lineal
│   ├── pedir_datos()           Input de puntos (x, y)
│   ├── mostrar_tabla()         Tabla comparativa + error
│   ├── graficar()              Gráfico matplotlib
│   └── predecir()              Estimación en x nuevo
│
├── Resolvers (lógica matemática)
│   ├── resolver_caso1()        f(x) = a·e^(b·x + c·x²)
│   ├── resolver_caso2()        f(x) = a·cos(b·x + π)
│   ├── resolver_caso3()        f(x) = a/(b+x)
│   └── resolver_caso4()        f(x) = a·x/(1+b·e^(c·x))
│
├── Funciones interactivas
│   ├── caso1_interactivo()     — caso4_interactivo()
│   ├── aproximacion_no_lineal()  Submenú 4 modelos
│   └── aproximacion_discreta()   Lineal con base del usuario
│
├── Ejemplos (datos hardcodeados)
│   ├── ejemplo_discreta()      Soplete con Chebyshev
│   ├── ejemplo_caso1() — ejemplo_caso4()
│   └── ejemplo_ingenieria()    Consumo servidor (11 puntos)
│
└── menu()                      Menú principal