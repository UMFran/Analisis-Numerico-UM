# Analisis-Numerico-UM
Repositorio hecho para desarrollar el temario de Aproximación por Mínimos Cuadrados de la asignatura de Análisis Numérico en la Universidad de Mendoza en 3er año de Ingeniería en Informática

# Aproximación por Mínimos Cuadrados

**Proyecto Informático — Análisis Numérico 2026**  
Facultad de Ingeniería — Universidad de Mendoza

---

## Autores

| Nombre | Carrera |
|--------|---------|
| Francisco Martin Gallardo | Ingeniería en Sistemas |
| Santiago Ariel Martinez | Ingeniería en Sistemas |

**Docente:** Bioing. Emiliano Aparicio  
**Contacto:** emiliano.aparicio@um.edu.ar

---

## Descripción

Este proyecto implementa una herramienta en Python para resolver problemas de **aproximación funcional por el método de los mínimos cuadrados**. El programa permite al usuario ingresar su propia función o sus propios datos y obtiene la mejor aproximación posible dentro de un subespacio dado.

Se cubren tres casos:

- **Aproximación continua:** dada una función f(x) conocida, se encuentra la mejor función aproximante φ*(x) en un intervalo [a, b].
- **Aproximación discreta:** dado un conjunto de puntos medidos (xk, yk), se encuentra la función que mejor los representa.
- **Aproximación discreta no lineal:** cuando los datos responden a una ley exponencial de la forma p(x) = a · e^(m·x), se aplica una transformación logarítmica para reducirlo al caso lineal.

En los tres casos el programa calcula el error del método y muestra un gráfico comparativo.

---

## Fundamento matemático

El método de los mínimos cuadrados busca el vector φ* en un subespacio S (de dimensión finita con base B = {φ₁, φ₂, ..., φₙ}) que minimiza la distancia al elemento u que se quiere aproximar.

Esto se reduce a resolver el **sistema de ecuaciones normales**:

```
A · X = B
```

donde:

```
A[j][i] = <φᵢ, φⱼ>    (productos escalares entre las funciones base)
B[j]    = <u, φⱼ>     (productos escalares de u con cada función base)
X       = [a₁, ..., aₙ]   (incógnitas: los coeficientes)
```

La matriz A es **simétrica y definida positiva** (llamada "matriz normal"), lo que garantiza solución única.

La mejor aproximación resulta:

```
φ*(x) = a₁·φ₁(x) + a₂·φ₂(x) + ... + aₙ·φₙ(x)
```

El error del método se calcula con:

```
E = ||u - φ*|| = sqrt( ||u||² - Σᵢ aᵢ · <u, φᵢ> )
```

### Producto escalar según el caso

| Caso | Producto escalar |
|------|-----------------|
| Continuo | `<f, g> = ∫ₐᵇ f(x)·g(x) dx` |
| Discreto | `<f, g> = Σₖ f(xₖ)·g(xₖ)` |

---

## Requisitos

- Python 3.x
- numpy
- scipy
- matplotlib

Instalación de dependencias:

```bash
pip install numpy scipy matplotlib
```

---

## Ejecución

```bash
python minimos_cuadrados.py
```

Al ejecutar aparece el siguiente menú:

```
=======================================================
  MÍNIMOS CUADRADOS — Análisis Numérico UM 2026
=======================================================
  1. Aproximación continua
  2. Aproximación discreta
  3. Aproximación discreta no lineal  [p(x) = a·e^(m·x)]
  0. Salir
```

---

## Uso de cada opción

### Opción 1 — Aproximación continua

El programa pide:

1. La función f(x) a aproximar
2. El intervalo [a, b]
3. La cantidad de funciones base y cada una de ellas

Ejemplo de entrada para aproximar sen(x) con la base {x, x³, x⁵} en [-1, 1]:

```
Función f(x) a aproximar: sin(x)
Límite inferior del intervalo (a): -1
Límite superior del intervalo (b): 1
¿Cuántas funciones tiene la base? 3
phi1(x) = x
phi2(x) = x**3
phi3(x) = x**5
```

Salida esperada:

```
f*(x) = (0.99998)·x + (-0.16652)·x³ + (0.00802)·x⁵
Error del método: E = 0.0000026237
```

### Opción 2 — Aproximación discreta

El programa pide:

1. La cantidad de puntos de datos
2. Cada par (x, y)
3. La cantidad de funciones base y cada una de ellas
4. Opcionalmente, un valor de x para estimar

Ejemplo de entrada (datos del soplete):

```
¿Cuántos puntos de datos tenés? 5
x1: 1    y1: 0.046
x2: 2    y2: 0.059
x3: 3    y3: 0.072
x4: 4    y4: 0.084
x5: 5    y5: 0.100
¿Cuántas funciones tiene la base? 3
phi1(x) = 1
phi2(x) = x
phi3(x) = 2*x**2-1
¿Querés estimar para algún x? (s/n): s
x = 2.5
```

### Opción 3 — Aproximación discreta no lineal

El programa pide únicamente los puntos (x, y). La transformación logarítmica y la resolución son automáticas. Al finalizar muestra los parámetros a y m de la función p(x) = a · e^(m·x).

```
¿Cuántos puntos de datos tenés? 4
x1: 1    y1: 7
x2: 2    y2: 11
x3: 3    y3: 17
x4: 4    y4: 27
```

Salida esperada:

```
p(x) = 4.467993 * e^(0.448510 * x)
Error del método: E = 0.20973276
```

---

## Funciones que se pueden usar en las expresiones

Al ingresar funciones por teclado se pueden usar:

| Expresión | Significado |
|-----------|-------------|
| `sin(x)` | seno |
| `cos(x)` | coseno |
| `tan(x)` | tangente |
| `exp(x)` | exponencial eˣ |
| `log(x)` | logaritmo natural |
| `sqrt(x)` | raíz cuadrada |
| `pi` | número π |
| `e` | número e |
| `x**2` | potencia (x²) |

---

## Estructura del código

```
minimos_cuadrados.py
│
├── evaluar(expresion, x)          → evalúa un string matemático en x
├── crear_funcion(expresion)       → convierte un string en función de Python
├── integrar(f, g, a, b)           → producto escalar continuo (Gauss-Legendre)
├── producto_discreto(lista_f, lista_g) → producto escalar discreto
├── resolver_sistema(A, B)         → muestra y resuelve A·X = B
│
├── aproximacion_continua()        → caso continuo interactivo
├── aproximacion_discreta()        → caso discreto interactivo
├── aproximacion_no_lineal()       → caso no lineal interactivo
│
└── menu()                         → menú principal
```

---

## Ejemplo de resultados

### Aproximación continua: sen(x) con base {x, x³, x⁵}

```
x        sen(x)       f*(x)        dif
-1.0000  -0.841471   -0.841478    0.000007
-0.6667  -0.618370   -0.618372    0.000002
 0.0000   0.000000    0.000000    0.000000
 0.6667   0.618370    0.618372    0.000002
 1.0000   0.841471    0.841478    0.000007

Error del método: E = 0.0000026237
```

La base {x, x³, x⁵} aprovecha que sen(x) es una función impar, por eso el error es tan pequeño.

### Aproximación no lineal: p(x) = a · e^(m·x)

```
x      y real     p(x)          dif
1.0      7.0     6.996774      0.003226
2.0     11.0    10.956786      0.043214
3.0     17.0    17.158073      0.158073
4.0     27.0    26.869146      0.130854
```

---

## Fecha de entrega

Antes del **31 de agosto de 2026**, con presentación virtual en horario de consulta acordado por correo con el docente.