"""
APROXIMACIÓN POR MÍNIMOS CUADRADOS
Análisis Numérico — Universidad de Mendoza 2026
Proyecto Informático
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


# ================================================================
# HERRAMIENTAS BASE
# ================================================================

def evaluar(expresion, x):
    """
    Evalúa una expresión matemática ingresada por el usuario.
    Podés usar: sin, cos, tan, exp, log, sqrt, pi, e, x
    """
    return float(eval(expresion, {
        "x": x,
        "sin": np.sin,  "cos": np.cos,  "tan": np.tan,
        "exp": np.exp,  "log": np.log,  "sqrt": np.sqrt,
        "pi":  np.pi,   "e":   np.e,    "abs":  abs
    }))


def crear_funcion(expresion):
    """
    Convierte un string como "sin(x)" en una función de Python.
    """
    def f(x):
        return evaluar(expresion, x)
    return f


def integrar(f, g, a, b):
    """
    Producto escalar continuo: integral de f(x)*g(x) en [a, b]
    Usa cuadratura de Gauss-Legendre (scipy.integrate.quad).
    """
    resultado, _ = integrate.quad(lambda x: f(x) * g(x), a, b)
    return resultado


def producto_discreto(lista_f, lista_g):
    """
    Producto escalar discreto: suma de f[k] * g[k]
    """
    suma = 0.0
    for k in range(len(lista_f)):
        suma += lista_f[k] * lista_g[k]
    return suma


def resolver_sistema(A, B):
    """
    Muestra y resuelve el sistema A * X = B.
    Retorna los coeficientes a1, a2, ..., an.
    """
    print("\n  Matriz A:")
    for fila in A:
        print("   ", [round(v, 5) for v in fila])
    print("\n  Vector B:")
    print("   ", [round(v, 5) for v in B])

    coefs = np.linalg.solve(np.array(A), np.array(B))

    print("\n  Coeficientes:")
    for i, c in enumerate(coefs):
        print(f"    a{i+1} = {c:.8f}")

    return coefs


# ================================================================
# APROXIMACIÓN CONTINUA — el usuario ingresa todo
# ================================================================

def aproximacion_continua():
    print("\n" + "=" * 55)
    print("  APROXIMACIÓN CONTINUA")
    print("  Podés usar: sin, cos, tan, exp, log, sqrt, pi, e")
    print("=" * 55)

    # ── El usuario ingresa la función y el intervalo ─────────────
    expr_f = input("\n  Función f(x) a aproximar: ")
    a = float(input("  Límite inferior del intervalo (a): "))
    b = float(input("  Límite superior del intervalo (b): "))

    f = crear_funcion(expr_f)

    # ── El usuario ingresa las funciones base ────────────────────
    n = int(input("  ¿Cuántas funciones tiene la base? "))
    base        = []
    nombres_base = []
    for i in range(n):
        expr = input(f"  phi{i+1}(x) = ")
        base.append(crear_funcion(expr))
        nombres_base.append(expr)

    # ── Armamos la matriz A ──────────────────────────────────────
    print("\n  Armando sistema de ecuaciones normales...")
    A = []
    for j in range(n):
        fila = []
        for i in range(n):
            fila.append(integrar(base[i], base[j], a, b))
        A.append(fila)

    # ── Armamos el vector B ──────────────────────────────────────
    B = []
    for j in range(n):
        B.append(integrar(f, base[j], a, b))

    # ── Resolvemos ───────────────────────────────────────────────
    coefs = resolver_sistema(A, B)

    # ── Función aproximada f*(x) ─────────────────────────────────
    def f_aprox(x):
        resultado = 0.0
        for i in range(n):
            resultado += coefs[i] * base[i](x)
        return resultado

    # ── Mostramos la expresión de f*(x) ─────────────────────────
    terminos = []
    for i in range(n):
        terminos.append(f"({coefs[i]:.5f})·{nombres_base[i]}")
    print("\n  f*(x) = " + " + ".join(terminos))

    # ── Error del método ─────────────────────────────────────────
    norma_cuad = integrar(f, f, a, b)
    correccion = 0.0
    for i in range(n):
        correccion += coefs[i] * integrar(f, base[i], a, b)
    E = np.sqrt(abs(norma_cuad - correccion))
    print(f"\n  Error del método: E = {E:.10f}")

    # ── Tabla de valores ─────────────────────────────────────────
    print("\n  Tabla de valores:")
    print(f"  {'x':>8}   {'f(x)':>12}   {'f*(x)':>12}   {'dif':>12}")
    paso = (b - a) / 6
    x_tabla = [a + i * paso for i in range(7)]
    for x in x_tabla:
        fx   = f(x)
        fapr = f_aprox(x)
        print(f"  {x:>8.4f}   {fx:>12.6f}   {fapr:>12.6f}   {abs(fx-fapr):>12.6f}")

    # ── Gráfico ──────────────────────────────────────────────────
    x_vals  = np.linspace(a, b, 300)
    y_orig  = [f(x) for x in x_vals]
    y_aprox = [f_aprox(x) for x in x_vals]

    plt.figure(figsize=(9, 5))
    plt.plot(x_vals, y_orig,  "b-",  linewidth=2, label=f"f(x) = {expr_f}")
    plt.plot(x_vals, y_aprox, "r--", linewidth=2, label="f*(x) = aproximación")
    plt.title(f"Aproximación Continua  —  Error = {E:.8f}")
    plt.xlabel("x"); plt.ylabel("y")
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


# ================================================================
# APROXIMACIÓN DISCRETA — el usuario ingresa la tabla de puntos
# ================================================================

def aproximacion_discreta():
    print("\n" + "=" * 55)
    print("  APROXIMACIÓN DISCRETA")
    print("  Podés usar: sin, cos, tan, exp, log, sqrt, pi, e")
    print("=" * 55)

    # ── El usuario ingresa los puntos (xk, yk) ──────────────────
    m = int(input("\n  ¿Cuántos puntos de datos tenés? "))
    x_datos = []
    y_datos = []
    print("  Ingresá los pares (x, y):")
    for k in range(m):
        x = float(input(f"    x{k+1}: "))
        y = float(input(f"    y{k+1}: "))
        x_datos.append(x)
        y_datos.append(y)

    # ── El usuario ingresa las funciones base ────────────────────
    n = int(input("\n  ¿Cuántas funciones tiene la base? "))
    base         = []
    nombres_base = []
    for i in range(n):
        expr = input(f"  phi{i+1}(x) = ")
        base.append(crear_funcion(expr))
        nombres_base.append(expr)

    # ── Evaluamos cada función base en los puntos de datos ───────
    base_vals = []
    for i in range(n):
        vals = []
        for k in range(m):
            vals.append(base[i](x_datos[k]))
        base_vals.append(vals)

    # ── Armamos la matriz A y el vector B ────────────────────────
    print("\n  Armando sistema de ecuaciones normales...")
    A = []
    for j in range(n):
        fila = []
        for i in range(n):
            fila.append(producto_discreto(base_vals[i], base_vals[j]))
        A.append(fila)

    B = []
    for j in range(n):
        B.append(producto_discreto(y_datos, base_vals[j]))

    # ── Resolvemos ───────────────────────────────────────────────
    coefs = resolver_sistema(A, B)

    # ── Función aproximada f*(x) ─────────────────────────────────
    def f_aprox(x):
        resultado = 0.0
        for i in range(n):
            resultado += coefs[i] * base[i](x)
        return resultado

    terminos = []
    for i in range(n):
        terminos.append(f"({coefs[i]:.5f})·{nombres_base[i]}")
    print("\n  f*(x) = " + " + ".join(terminos))

    # ── Error del método ─────────────────────────────────────────
    norma_cuad = producto_discreto(y_datos, y_datos)
    correccion = 0.0
    for i in range(n):
        correccion += coefs[i] * producto_discreto(y_datos, base_vals[i])
    E = np.sqrt(abs(norma_cuad - correccion))
    print(f"\n  Error del método: E = {E:.10f}")

    # ── Tabla comparativa ────────────────────────────────────────
    print("\n  Tabla comparativa:")
    print(f"  {'x':>8}   {'y real':>10}   {'f*(x)':>10}   {'dif':>10}")
    for xi, yi in zip(x_datos, y_datos):
        fi = f_aprox(xi)
        print(f"  {xi:>8.4f}   {yi:>10.6f}   {fi:>10.6f}   {abs(yi-fi):>10.6f}")

    # ── Predicción en un punto nuevo ─────────────────────────────
    pred = input("\n  ¿Querés estimar para algún x? (s/n): ").strip().lower()
    if pred == "s":
        x_pred = float(input("  x = "))
        print(f"  f*({x_pred}) = {f_aprox(x_pred):.8f}")

    # ── Gráfico ──────────────────────────────────────────────────
    margen = (max(x_datos) - min(x_datos)) * 0.1
    x_plot = np.linspace(min(x_datos) - margen, max(x_datos) + margen, 300)
    y_plot = [f_aprox(x) for x in x_plot]

    plt.figure(figsize=(9, 5))
    plt.scatter(x_datos, y_datos, color="blue", s=100, zorder=5, label="Datos originales")
    plt.plot(x_plot, y_plot, "r-", linewidth=2, label="f*(x) = aproximación")
    plt.title(f"Aproximación Discreta  —  Error = {E:.6f}")
    plt.xlabel("x"); plt.ylabel("y")
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


# ================================================================
# APROXIMACIÓN DISCRETA NO LINEAL — el usuario ingresa los puntos
# Asume la forma p(x) = a * e^(m*x) y linealiza con ln
# ================================================================

def aproximacion_no_lineal():
    print("\n" + "=" * 55)
    print("  APROXIMACIÓN DISCRETA NO LINEAL")
    print("  Forma: p(x) = a * e^(m*x)")
    print("  Truco: aplicamos ln → queda lineal")
    print("=" * 55)

    # ── El usuario ingresa los puntos ────────────────────────────
    m_pts = int(input("\n  ¿Cuántos puntos de datos tenés? "))
    x_datos = []
    y_datos = []
    print("  Ingresá los pares (x, y):")
    for k in range(m_pts):
        x = float(input(f"    x{k+1}: "))
        y = float(input(f"    y{k+1}: "))
        x_datos.append(x)
        y_datos.append(y)

    # ── Linearización: ln(y) = ln(a) + m*x ──────────────────────
    print("\n  Linearización: aplicando ln(y)...")
    y_ln = []
    for yi in y_datos:
        y_ln.append(np.log(yi))

    print(f"\n  {'x':>8}   {'y':>8}   {'ln(y)':>10}")
    for xi, yi, lyi in zip(x_datos, y_datos, y_ln):
        print(f"  {xi:>8.4f}   {yi:>8.4f}   {lyi:>10.6f}")

    # ── Base lineal: {1, x} ──────────────────────────────────────
    def phi1(x): return 1.0
    def phi2(x): return x

    base = [phi1, phi2]
    n    = len(base)

    base_vals = []
    for i in range(n):
        vals = []
        for k in range(m_pts):
            vals.append(base[i](x_datos[k]))
        base_vals.append(vals)

    # ── Sistema en el espacio logarítmico ────────────────────────
    print("\n  Armando sistema en el espacio logarítmico...")
    A = []
    for j in range(n):
        fila = []
        for i in range(n):
            fila.append(producto_discreto(base_vals[i], base_vals[j]))
        A.append(fila)

    B = []
    for j in range(n):
        B.append(producto_discreto(y_ln, base_vals[j]))

    coefs = resolver_sistema(A, B)
    h = coefs[0]
    k = coefs[1]

    # ── Deshacemos la transformación ─────────────────────────────
    a_val = np.exp(h)
    m_val = k

    print(f"\n  h = ln(a) = {h:.8f}  →  a = e^h = {a_val:.8f}")
    print(f"  k = m     = {m_val:.8f}")
    print(f"\n  p(x) = {a_val:.6f} * e^({m_val:.6f} * x)")

    def p(x):
        return a_val * np.exp(m_val * x)

    # ── Error ────────────────────────────────────────────────────
    suma_dif_cuad = 0.0
    for k_idx in range(m_pts):
        suma_dif_cuad += (y_datos[k_idx] - p(x_datos[k_idx])) ** 2
    E = np.sqrt(suma_dif_cuad)
    print(f"\n  Error del método: E = {E:.8f}")

    # ── Tabla comparativa ────────────────────────────────────────
    print("\n  Verificación:")
    print(f"  {'x':>8}   {'y real':>8}   {'p(x)':>12}   {'dif':>10}")
    for xi, yi in zip(x_datos, y_datos):
        print(f"  {xi:>8.4f}   {yi:>8.4f}   {p(xi):>12.6f}   {abs(yi-p(xi)):>10.6f}")

    # ── Predicción ───────────────────────────────────────────────
    pred = input("\n  ¿Querés estimar para algún x? (s/n): ").strip().lower()
    if pred == "s":
        x_pred = float(input("  x = "))
        print(f"  p({x_pred}) = {p(x_pred):.8f}")

    # ── Gráfico ──────────────────────────────────────────────────
    margen = (max(x_datos) - min(x_datos)) * 0.1
    x_plot = np.linspace(min(x_datos) - margen, max(x_datos) + margen, 300)
    y_plot = [p(x) for x in x_plot]

    plt.figure(figsize=(9, 5))
    plt.scatter(x_datos, y_datos, color="blue", s=120, zorder=5, label="Datos originales")
    plt.plot(x_plot, y_plot, "r-", linewidth=2,
             label=f"p(x) = {a_val:.4f}·e^({m_val:.4f}·x)")
    plt.title(f"No Lineal — Ajuste exponencial  —  Error = {E:.6f}")
    plt.xlabel("x"); plt.ylabel("y")
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


# ================================================================
# EJEMPLOS PREDEFINIDOS (los de la documentación)
# ================================================================

def ejemplo_continua():
    """
    Ejemplo continuo: aproximar sen(x) con base {x, x³, x⁵} en [-1, 1].
    Los datos ya están cargados, no hay que ingresar nada.
    """
    print("\n" + "=" * 55)
    print("  EJEMPLO — Aproximación Continua")
    print("  f(x) = sen(x)  |  base {x, x³, x⁵}  |  [-1, 1]")
    print("=" * 55)

    def f(x):     return np.sin(x)
    def phi1(x):  return x
    def phi2(x):  return x ** 3
    def phi3(x):  return x ** 5

    base         = [phi1, phi2, phi3]
    nombres_base = ["x", "x**3", "x**5"]
    n = len(base)
    a, b = -1.0, 1.0

    print("\n  Armando sistema de ecuaciones normales...")
    A = []
    for j in range(n):
        fila = []
        for i in range(n):
            fila.append(integrar(base[i], base[j], a, b))
        A.append(fila)

    B = []
    for j in range(n):
        B.append(integrar(f, base[j], a, b))

    coefs = resolver_sistema(A, B)
    a1, a2, a3 = coefs[0], coefs[1], coefs[2]

    def f_aprox(x):
        return a1 * phi1(x) + a2 * phi2(x) + a3 * phi3(x)

    terminos = []
    for i in range(n):
        terminos.append(f"({coefs[i]:.5f})·{nombres_base[i]}")
    print("\n  f*(x) = " + " + ".join(terminos))

    norma_cuad = integrar(f, f, a, b)
    correccion = 0.0
    for i in range(n):
        correccion += coefs[i] * integrar(f, base[i], a, b)
    E = np.sqrt(abs(norma_cuad - correccion))
    print(f"\n  Error del método: E = {E:.10f}")

    print("\n  Tabla de valores:")
    print(f"  {'x':>8}   {'sen(x)':>12}   {'f*(x)':>12}   {'dif':>12}")
    for x in [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]:
        fx   = f(x)
        fapr = f_aprox(x)
        print(f"  {x:>8.4f}   {fx:>12.6f}   {fapr:>12.6f}   {abs(fx-fapr):>12.8f}")

    x_vals  = np.linspace(a, b, 300)
    y_orig  = [f(x) for x in x_vals]
    y_aprox = [f_aprox(x) for x in x_vals]

    plt.figure(figsize=(9, 5))
    plt.plot(x_vals, y_orig,  "b-",  linewidth=2, label="f(x) = sen(x)")
    plt.plot(x_vals, y_aprox, "r--", linewidth=2, label="f*(x) con base {x, x³, x⁵}")
    plt.title(f"Ejemplo Continuo — Error = {E:.8f}")
    plt.xlabel("x"); plt.ylabel("y")
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def ejemplo_discreta():
    """
    Ejemplo discreto: tiempo de corte con soplete.
    Base de Chebyshev: {1, x, 2x²-1}.
    """
    print("\n" + "=" * 55)
    print("  EJEMPLO — Aproximación Discreta")
    print("  Soplete: espesor (pulg) vs tiempo de corte (min)")
    print("  Base: polinomios de Chebyshev {1, x, 2x²-1}")
    print("=" * 55)

    x_datos = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_datos = [0.046, 0.059, 0.072, 0.084, 0.100]

    print("\n  Datos:")
    print(f"  {'E (pulg)':>10}   {'t (min)':>10}")
    for xi, yi in zip(x_datos, y_datos):
        print(f"  {xi:>10.1f}   {yi:>10.3f}")

    def phi1(x): return 1.0
    def phi2(x): return x
    def phi3(x): return 2 * x**2 - 1

    base         = [phi1, phi2, phi3]
    nombres_base = ["1", "x", "2*x**2-1"]
    n = len(base)
    m = len(x_datos)

    base_vals = []
    for i in range(n):
        vals = []
        for k in range(m):
            vals.append(base[i](x_datos[k]))
        base_vals.append(vals)

    print("\n  Armando sistema de ecuaciones normales...")
    A = []
    for j in range(n):
        fila = []
        for i in range(n):
            fila.append(producto_discreto(base_vals[i], base_vals[j]))
        A.append(fila)

    B = []
    for j in range(n):
        B.append(producto_discreto(y_datos, base_vals[j]))

    coefs = resolver_sistema(A, B)
    a1, a2, a3 = coefs[0], coefs[1], coefs[2]

    def f_aprox(x):
        return a1 * phi1(x) + a2 * phi2(x) + a3 * phi3(x)

    terminos = []
    for i in range(n):
        terminos.append(f"({coefs[i]:.5f})·{nombres_base[i]}")
    print("\n  f*(x) = " + " + ".join(terminos))

    norma_cuad = producto_discreto(y_datos, y_datos)
    correccion = 0.0
    for i in range(n):
        correccion += coefs[i] * producto_discreto(y_datos, base_vals[i])
    E = np.sqrt(abs(norma_cuad - correccion))
    print(f"\n  Error del método: E = {E:.10f}")

    print("\n  Tabla comparativa:")
    print(f"  {'x':>6}   {'y real':>8}   {'f*(x)':>10}   {'dif':>10}")
    for xi, yi in zip(x_datos, y_datos):
        fi = f_aprox(xi)
        print(f"  {xi:>6.1f}   {yi:>8.5f}   {fi:>10.6f}   {abs(yi-fi):>10.6f}")

    print(f"\n  Predicción para E = 2.5 pulg:  t = {f_aprox(2.5):.6f} min")

    margen = 0.3
    x_plot = np.linspace(min(x_datos) - margen, max(x_datos) + margen, 300)
    y_plot = [f_aprox(x) for x in x_plot]

    plt.figure(figsize=(9, 5))
    plt.scatter(x_datos, y_datos, color="blue", s=100, zorder=5, label="Datos reales")
    plt.plot(x_plot, y_plot, "r-", linewidth=2, label="f*(x) — Chebyshev")
    plt.title(f"Ejemplo Discreto (Soplete) — Error = {E:.6f}")
    plt.xlabel("Espesor (pulgadas)"); plt.ylabel("Tiempo (min)")
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def ejemplo_no_lineal():
    """
    Ejemplo no lineal: ajuste de p(x) = a * e^(m*x).
    Se lineariza con ln y se resuelve como caso discreto lineal.
    """
    print("\n" + "=" * 55)
    print("  EJEMPLO — Aproximación Discreta No Lineal")
    print("  Forma: p(x) = a * e^(m*x)")
    print("=" * 55)

    x_datos = [1.0, 2.0, 3.0, 4.0]
    y_datos = [7.0, 11.0, 17.0, 27.0]

    print("\n  Datos originales:")
    print(f"  {'x':>5}   {'y':>6}")
    for xi, yi in zip(x_datos, y_datos):
        print(f"  {xi:>5.1f}   {yi:>6.1f}")

    print("\n  Linearización: aplicando ln(y)...")
    y_ln = []
    for yi in y_datos:
        y_ln.append(np.log(yi))

    print(f"\n  {'x':>5}   {'y':>6}   {'ln(y)':>10}")
    for xi, yi, lyi in zip(x_datos, y_datos, y_ln):
        print(f"  {xi:>5.1f}   {yi:>6.1f}   {lyi:>10.6f}")

    def phi1(x): return 1.0
    def phi2(x): return x

    base = [phi1, phi2]
    n    = len(base)
    m_pts = len(x_datos)

    base_vals = []
    for i in range(n):
        vals = []
        for k in range(m_pts):
            vals.append(base[i](x_datos[k]))
        base_vals.append(vals)

    print("\n  Armando sistema en el espacio logarítmico...")
    A = []
    for j in range(n):
        fila = []
        for i in range(n):
            fila.append(producto_discreto(base_vals[i], base_vals[j]))
        A.append(fila)

    B = []
    for j in range(n):
        B.append(producto_discreto(y_ln, base_vals[j]))

    coefs = resolver_sistema(A, B)
    h = coefs[0]
    k = coefs[1]

    a_val = np.exp(h)
    m_val = k

    print(f"\n  h = ln(a) = {h:.8f}  →  a = e^h = {a_val:.8f}")
    print(f"  k = m     = {m_val:.8f}")
    print(f"\n  p(x) = {a_val:.6f} * e^({m_val:.6f} * x)")

    def p(x):
        return a_val * np.exp(m_val * x)

    suma_dif_cuad = 0.0
    for k_idx in range(m_pts):
        suma_dif_cuad += (y_datos[k_idx] - p(x_datos[k_idx])) ** 2
    E = np.sqrt(suma_dif_cuad)
    print(f"\n  Error del método: E = {E:.8f}")

    print("\n  Verificación:")
    print(f"  {'x':>5}   {'y real':>8}   {'p(x)':>12}   {'dif':>10}")
    for xi, yi in zip(x_datos, y_datos):
        print(f"  {xi:>5.1f}   {yi:>8.1f}   {p(xi):>12.6f}   {abs(yi-p(xi)):>10.6f}")

    x_plot = np.linspace(0.5, 4.5, 300)
    y_plot = [p(x) for x in x_plot]

    plt.figure(figsize=(9, 5))
    plt.scatter(x_datos, y_datos, color="blue", s=120, zorder=5, label="Datos originales")
    plt.plot(x_plot, y_plot, "r-", linewidth=2,
             label=f"p(x) = {a_val:.4f}·e^({m_val:.4f}·x)")
    plt.title(f"Ejemplo No Lineal — Error = {E:.6f}")
    plt.xlabel("x"); plt.ylabel("y")
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


# ================================================================
# MENÚ
# ================================================================

def menu():
    while True:
        print("\n" + "=" * 55)
        print("  MÍNIMOS CUADRADOS — Análisis Numérico UM 2026")
        print("=" * 55)
        print("  1. Aproximación continua")
        print("  2. Aproximación discreta")
        print("  3. Aproximación discreta no lineal  [p(x) = a·e^(m·x)]")
        print("  4. Ver ejemplos de la documentación")
        print("  0. Salir")

        opcion = input("\n  Opción: ").strip()

        if opcion == "1":
            aproximacion_continua()
        elif opcion == "2":
            aproximacion_discreta()
        elif opcion == "3":
            aproximacion_no_lineal()
        elif opcion == "4":
            while True:
                print("\n" + "-" * 55)
                print("  EJEMPLOS")
                print("-" * 55)
                print("  1. Continua   — sen(x) con base {x, x³, x⁵}")
                print("  2. Discreta   —  caso soplete")
                print("  3. No lineal  — p(x) = a·e^(m·x)")
                print("  0. Volver al menú principal")

                sub = input("\n  Ejemplo: ").strip()

                if sub == "1":
                    ejemplo_continua()
                elif sub == "2":
                    ejemplo_discreta()
                elif sub == "3":
                    ejemplo_no_lineal()
                elif sub == "0":
                    break
                else:
                    print("  Opción no válida.")

                input("\n  [Enter para continuar...]")

        elif opcion == "0":
            print("\n  Gracias por usar el programa.\n")
            break
        else:
            print("  Opción no válida.")

        if opcion != "4":
            input("\n  [Enter para continuar...]")


if __name__ == "__main__":
    menu()