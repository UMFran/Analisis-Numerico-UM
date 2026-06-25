"""
APROXIMACIÓN POR MÍNIMOS CUADRADOS — Discreta y No Lineal
Análisis Numérico — Universidad de Mendoza 2026
Proyecto Informático

Autores:
    Francisco Martin Gallardo
    Santiago Ariel Martinez

Para ejecutar:
    pip install numpy matplotlib
    python minimos_cuadrados.py
"""

import numpy as np
import matplotlib.pyplot as plt


# ================================================================
# HERRAMIENTAS BASE
# ================================================================

def evaluar(expresion, x):
    """Evalúa un string matemático con variable x."""
    return float(eval(expresion, {
        "x": x, "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
        "pi": np.pi, "e": np.e, "abs": abs
    }))

def crear_funcion(expresion):
    """Convierte un string como 'sin(x)' en función de Python."""
    def f(x):
        return evaluar(expresion, x)
    return f

def producto_discreto(lista_f, lista_g):
    """Producto escalar discreto: suma de f[k] * g[k]"""
    suma = 0.0
    for k in range(len(lista_f)):
        suma += lista_f[k] * lista_g[k]
    return suma

def resolver_sistema(A, B):
    """Muestra y resuelve A * X = B con numpy."""
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
# UTILIDADES COMPARTIDAS PARA LOS CASOS NO LINEALES
# ================================================================

def pedir_datos():
    """Pide los puntos (xk, yk) al usuario."""
    m = int(input("\n  ¿Cuántos puntos de datos tenés? "))
    x_datos, y_datos = [], []
    print("  Ingresá los pares (x, y):")
    for k in range(m):
        x = float(input(f"    x{k+1}: "))
        y = float(input(f"    y{k+1}: "))
        x_datos.append(x)
        y_datos.append(y)
    return x_datos, y_datos

def mostrar_tabla(x_datos, y_datos, p):
    """Muestra tabla comparativa y calcula el error."""
    E = np.sqrt(sum((yi - p(xi))**2 for xi, yi in zip(x_datos, y_datos)))
    print(f"\n  Error del método: E = {E:.8f}")
    print(f"\n  {'x':>8}   {'y real':>10}   {'p(x)':>12}   {'dif':>10}")
    for xi, yi in zip(x_datos, y_datos):
        print(f"  {xi:>8.4f}   {yi:>10.4f}   {p(xi):>12.6f}   {abs(yi-p(xi)):>10.6f}")
    return E

def graficar(x_datos, y_datos, p, titulo, label_p, xlabel="x", ylabel="y"):
    """Grafica los datos y la curva ajustada."""
    margen = (max(x_datos) - min(x_datos)) * 0.1
    x_plot = np.linspace(min(x_datos) - margen, max(x_datos) + margen, 300)
    y_plot = [p(x) for x in x_plot]
    plt.figure(figsize=(9, 5))
    plt.scatter(x_datos, y_datos, color="blue", s=100, zorder=5, label="Datos originales")
    plt.plot(x_plot, y_plot, "r-", linewidth=2, label=label_p)
    plt.title(titulo); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.tight_layout(); plt.show()

def predecir(p):
    """Pregunta si el usuario quiere estimar para un x nuevo."""
    if input("\n  ¿Querés estimar para algún x? (s/n): ").strip().lower() == "s":
        xp = float(input("  x = "))
        print(f"  p({xp}) = {p(xp):.8f}")


# ================================================================
# CASO 1: f(x) = a * e^(b*x + c*x²)
# Linearización: ln(y) = ln(a) + b*x + c*x²
# Base: {1, x, x²}  con  z = ln(y)
# Recuperar: a = e^h,  b = k1,  c = k2
# ================================================================

def resolver_caso1(x_datos, y_datos, mostrar=True):
    z = [np.log(yi) for yi in y_datos]

    def phi1(x): return 1.0
    def phi2(x): return x
    def phi3(x): return x ** 2

    base = [phi1, phi2, phi3]
    n = 3
    base_vals = [[base[i](xk) for xk in x_datos] for i in range(n)]

    A = [[producto_discreto(base_vals[i], base_vals[j]) for i in range(n)] for j in range(n)]
    B = [producto_discreto(z, base_vals[j]) for j in range(n)]

    if mostrar:
        print("\n  Transformación: z = ln(y)")
        print(f"  {'x':>8}   {'y':>10}   {'z=ln(y)':>10}")
        for xi, yi, zi in zip(x_datos, y_datos, z):
            print(f"  {xi:>8.3f}   {yi:>10.4f}   {zi:>10.6f}")
        print("\n  Sistema de ecuaciones normales (base {1, x, x²}):")

    coefs = resolver_sistema(A, B)
    h, k1, k2 = coefs[0], coefs[1], coefs[2]

    a_val = np.exp(h)
    b_val, c_val = k1, k2

    print(f"\n  Recuperando parámetros:")
    print(f"    h  = ln(a) = {h:.8f}  →  a = e^h = {a_val:.8f}")
    print(f"    k1 = b     = {b_val:.8f}")
    print(f"    k2 = c     = {c_val:.8f}")
    print(f"\n  p(x) = {a_val:.5f} * e^({b_val:.5f}*x + {c_val:.5f}*x²)")

    def p(x):
        return a_val * np.exp(b_val * x + c_val * x ** 2)

    return p, a_val, b_val, c_val


def caso1_interactivo():
    print("\n" + "=" * 60)
    print("  CASO 1: f(x) = a * e^(b*x + c*x²)")
    print("  Linearización: ln(y) = ln(a) + b*x + c*x²")
    print("=" * 60)
    x_datos, y_datos = pedir_datos()
    p, a, b, c = resolver_caso1(x_datos, y_datos)
    E = mostrar_tabla(x_datos, y_datos, p)
    predecir(p)
    graficar(x_datos, y_datos, p,
             f"Caso 1 — Error = {E:.6f}",
             f"p(x) = {a:.4f}·e^({b:.4f}x + {c:.4f}x²)")


# ================================================================
# CASO 2: f(x) = a * cos(b*x + π)
# Simplificación: cos(bx + π) = -cos(bx)  →  f(x) = -a*cos(bx)
# Con b conocido: sistema 1×1 lineal en a
# ================================================================

def resolver_caso2(x_datos, y_datos, b_val, mostrar=True):
    g_vals = [-np.cos(b_val * xk) for xk in x_datos]

    A = [[producto_discreto(g_vals, g_vals)]]
    B = [producto_discreto(y_datos, g_vals)]

    if mostrar:
        print(f"\n  Simplificación: cos(b*x + π) = -cos(b*x)")
        print(f"  f(x) = a · (-cos({b_val}·x))")
        print(f"\n  {'x':>8}   {'y':>10}   {'g(x)=-cos(bx)':>14}")
        for xi, yi, gi in zip(x_datos, y_datos, g_vals):
            print(f"  {xi:>8.3f}   {yi:>10.4f}   {gi:>14.6f}")
        print("\n  Sistema de ecuaciones normales (1x1):")

    coefs = resolver_sistema(A, B)
    a_val = coefs[0]

    print(f"\n  a = {a_val:.8f}")
    print(f"\n  p(x) = {a_val:.5f} * cos({b_val}·x + π)")

    def p(x):
        return a_val * np.cos(b_val * x + np.pi)

    return p, a_val


def caso2_interactivo():
    print("\n" + "=" * 60)
    print("  CASO 2: f(x) = a * cos(b*x + π)")
    print("  Simplificación: = -a * cos(b*x)")
    print("  Con b conocido → sistema 1x1 lineal en a")
    print("  (b se estima del período: T = 2π/b  →  b = 2π/T)")
    print("=" * 60)
    x_datos, y_datos = pedir_datos()
    b_val = float(input("\n  Valor de b (estimado del período de los datos): "))
    p, a = resolver_caso2(x_datos, y_datos, b_val)
    E = mostrar_tabla(x_datos, y_datos, p)
    predecir(p)
    graficar(x_datos, y_datos, p,
             f"Caso 2 — Error = {E:.6f}",
             f"p(x) = {a:.4f}·cos({b_val}·x + π)")


# ================================================================
# CASO 3: f(x) = a / (b + x)
# Linearización: 1/y = b/a + (1/a)*x  →  z = h·1 + k·x
# Base: {1, x}  con  z = 1/y
# Recuperar: a = 1/k,  b = h/k
# ================================================================

def resolver_caso3(x_datos, y_datos, mostrar=True):
    z = [1.0 / yi for yi in y_datos]

    def phi1(x): return 1.0
    def phi2(x): return x

    base = [phi1, phi2]
    n = 2
    base_vals = [[base[i](xk) for xk in x_datos] for i in range(n)]

    A = [[producto_discreto(base_vals[i], base_vals[j]) for i in range(n)] for j in range(n)]
    B = [producto_discreto(z, base_vals[j]) for j in range(n)]

    if mostrar:
        print("\n  Transformación: z = 1/y")
        print(f"  {'x':>8}   {'y':>10}   {'z=1/y':>10}")
        for xi, yi, zi in zip(x_datos, y_datos, z):
            print(f"  {xi:>8.3f}   {yi:>10.4f}   {zi:>10.6f}")
        print("\n  Sistema de ecuaciones normales (base {1, x}):")

    coefs = resolver_sistema(A, B)
    h, k = coefs[0], coefs[1]

    a_val = 1.0 / k
    b_val = h / k

    print(f"\n  Recuperando parámetros:")
    print(f"    h = b/a = {h:.8f}")
    print(f"    k = 1/a = {k:.8f}  →  a = 1/k = {a_val:.8f}")
    print(f"    b = h/k = {b_val:.8f}")
    print(f"\n  p(x) = {a_val:.5f} / ({b_val:.5f} + x)")

    def p(x):
        return a_val / (b_val + x)

    return p, a_val, b_val


def caso3_interactivo():
    print("\n" + "=" * 60)
    print("  CASO 3: f(x) = a / (b + x)")
    print("  Linearización: 1/y = b/a + (1/a)*x")
    print("  Base del sistema: {1, x}  con  z = 1/y")
    print("=" * 60)
    x_datos, y_datos = pedir_datos()
    p, a, b = resolver_caso3(x_datos, y_datos)
    E = mostrar_tabla(x_datos, y_datos, p)
    predecir(p)
    graficar(x_datos, y_datos, p,
             f"Caso 3 — Error = {E:.6f}",
             f"p(x) = {a:.4f} / ({b:.4f} + x)")


# ================================================================
# CASO 4: f(x) = a*x / (1 + b*e^(c*x))
# Linearización: x/y = (1/a) + (b/a)*e^(c*x)  →  z = h·1 + k·e^(cx)
# Base: {1, e^(c*x)}  con  z = x/y  (c debe ser conocido)
# Recuperar: a = 1/h,  b = k/h
# ================================================================

def resolver_caso4(x_datos, y_datos, c_val, mostrar=True):
    z = [xk / yk for xk, yk in zip(x_datos, y_datos)]

    def phi1(x): return 1.0
    def phi2(x): return np.exp(c_val * x)

    base = [phi1, phi2]
    n = 2
    base_vals = [[base[i](xk) for xk in x_datos] for i in range(n)]

    A = [[producto_discreto(base_vals[i], base_vals[j]) for i in range(n)] for j in range(n)]
    B = [producto_discreto(z, base_vals[j]) for j in range(n)]

    if mostrar:
        print(f"\n  Transformación: z = x/y  (usando c = {c_val})")
        print(f"  {'x':>8}   {'y':>10}   {'z=x/y':>10}   {'e^(c*x)':>12}")
        for xi, yi, zi in zip(x_datos, y_datos, z):
            print(f"  {xi:>8.3f}   {yi:>10.4f}   {zi:>10.6f}   {np.exp(c_val*xi):>12.6f}")
        print(f"\n  Sistema de ecuaciones normales (base {{1, e^({c_val}*x)}}):")

    coefs = resolver_sistema(A, B)
    h, k = coefs[0], coefs[1]

    a_val = 1.0 / h
    b_val = k / h

    print(f"\n  Recuperando parámetros:")
    print(f"    h = 1/a = {h:.8f}  →  a = 1/h = {a_val:.8f}")
    print(f"    k = b/a = {k:.8f}  →  b = k/h = {b_val:.8f}")
    print(f"    c (dado) = {c_val}")
    print(f"\n  p(x) = {a_val:.5f}·x / (1 + {b_val:.5f}·e^({c_val}·x))")

    def p(x):
        return a_val * x / (1 + b_val * np.exp(c_val * x))

    return p, a_val, b_val


def caso4_interactivo():
    print("\n" + "=" * 60)
    print("  CASO 4: f(x) = a*x / (1 + b*e^(c*x))")
    print("  Linearización: x/y = (1/a) + (b/a)*e^(c*x)")
    print("  Base del sistema: {1, e^(c*x)}  (c debe ser conocido)")
    print("=" * 60)
    x_datos, y_datos = pedir_datos()
    c_val = float(input("\n  Valor de c (estimado de los datos): "))
    p, a, b = resolver_caso4(x_datos, y_datos, c_val)
    E = mostrar_tabla(x_datos, y_datos, p)
    predecir(p)
    graficar(x_datos, y_datos, p,
             f"Caso 4 — Error = {E:.6f}",
             f"p(x) = {a:.4f}·x / (1 + {b:.4f}·e^({c_val}·x))")


def aproximacion_no_lineal():
    """Submenú con los 4 modelos no lineales."""
    while True:
        print("\n" + "=" * 60)
        print("  APROXIMACIÓN DISCRETA NO LINEAL")
        print("=" * 60)
        print("  1. f(x) = a * e^(b*x + c*x²)        [Exp. cuadrática]")
        print("  2. f(x) = a * cos(b*x + π)           [Coseno]")
        print("  3. f(x) = a / (b + x)                [Hipérbola]")
        print("  4. f(x) = a*x / (1 + b*e^(c*x))     [Logística]")
        print("  0. Volver")

        sub = input("\n  Modelo: ").strip()
        if sub == "1":   caso1_interactivo()
        elif sub == "2": caso2_interactivo()
        elif sub == "3": caso3_interactivo()
        elif sub == "4": caso4_interactivo()
        elif sub == "0": break
        else: print("  Opción no válida.")
        if sub != "0":
            input("\n  [Enter para continuar...]")


# ================================================================
# APROXIMACIÓN DISCRETA — el usuario ingresa datos y base
# ================================================================

def aproximacion_discreta():
    print("\n" + "=" * 60)
    print("  APROXIMACIÓN DISCRETA")
    print("  Podés usar en las funciones: sin, cos, exp, log, x**n")
    print("=" * 60)

    m = int(input("\n  ¿Cuántos puntos de datos tenés? "))
    x_datos, y_datos = [], []
    print("  Ingresá los pares (x, y):")
    for k in range(m):
        x_datos.append(float(input(f"    x{k+1}: ")))
        y_datos.append(float(input(f"    y{k+1}: ")))

    n = int(input("\n  ¿Cuántas funciones tiene la base? "))
    base, nombres_base = [], []
    for i in range(n):
        expr = input(f"  phi{i+1}(x) = ")
        base.append(crear_funcion(expr))
        nombres_base.append(expr)

    base_vals = [[base[i](xk) for xk in x_datos] for i in range(n)]

    print("\n  Armando sistema de ecuaciones normales...")
    A = [[producto_discreto(base_vals[i], base_vals[j]) for i in range(n)] for j in range(n)]
    B = [producto_discreto(y_datos, base_vals[j]) for j in range(n)]

    coefs = resolver_sistema(A, B)

    def f_aprox(x):
        return sum(coefs[i] * base[i](x) for i in range(n))

    terminos = [f"({coefs[i]:.5f})·{nombres_base[i]}" for i in range(n)]
    print("\n  f*(x) = " + " + ".join(terminos))

    norma_cuad = producto_discreto(y_datos, y_datos)
    correccion = sum(coefs[i] * producto_discreto(y_datos, base_vals[i]) for i in range(n))
    E = np.sqrt(abs(norma_cuad - correccion))
    print(f"\n  Error del método: E = {E:.10f}")

    print("\n  Tabla comparativa:")
    print(f"  {'x':>8}   {'y real':>10}   {'f*(x)':>10}   {'dif':>10}")
    for xi, yi in zip(x_datos, y_datos):
        fi = f_aprox(xi)
        print(f"  {xi:>8.4f}   {yi:>10.6f}   {fi:>10.6f}   {abs(yi-fi):>10.6f}")

    if input("\n  ¿Querés estimar para algún x? (s/n): ").strip().lower() == "s":
        xp = float(input("  x = "))
        print(f"  f*({xp}) = {f_aprox(xp):.8f}")

    margen = (max(x_datos) - min(x_datos)) * 0.1
    x_plot = np.linspace(min(x_datos) - margen, max(x_datos) + margen, 300)
    y_plot = [f_aprox(x) for x in x_plot]

    plt.figure(figsize=(9, 5))
    plt.scatter(x_datos, y_datos, color="blue", s=100, zorder=5, label="Datos originales")
    plt.plot(x_plot, y_plot, "r-", linewidth=2, label="f*(x) = aproximación")
    plt.title(f"Aproximación Discreta  —  Error = {E:.6f}")
    plt.xlabel("x"); plt.ylabel("y")
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.tight_layout(); plt.show()


# ================================================================
# EJEMPLOS PREDEFINIDOS
# ================================================================

def ejemplo_discreta():
    """Soplete: datos de (espesor, tiempo) con base Chebyshev."""
    print("\n" + "=" * 60)
    print("  EJEMPLO — Discreta: tiempo de corte con soplete")
    print("  Base: polinomios de Chebyshev {1, x, 2x²-1}")
    print("=" * 60)

    x_datos = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_datos = [0.046, 0.059, 0.072, 0.084, 0.100]

    print("\n  Datos:")
    print(f"  {'E (pulg)':>10}   {'t (min)':>10}")
    for xi, yi in zip(x_datos, y_datos):
        print(f"  {xi:>10.1f}   {yi:>10.3f}")

    def phi1(x): return 1.0
    def phi2(x): return x
    def phi3(x): return 2 * x**2 - 1

    base = [phi1, phi2, phi3]
    nombres = ["1", "x", "2x²-1"]
    n, m = 3, len(x_datos)
    base_vals = [[base[i](xk) for xk in x_datos] for i in range(n)]

    A = [[producto_discreto(base_vals[i], base_vals[j]) for i in range(n)] for j in range(n)]
    B = [producto_discreto(y_datos, base_vals[j]) for j in range(n)]
    coefs = resolver_sistema(A, B)

    def f_aprox(x):
        return sum(coefs[i] * base[i](x) for i in range(n))

    terminos = [f"({coefs[i]:.5f})·{nombres[i]}" for i in range(n)]
    print("\n  f*(x) = " + " + ".join(terminos))

    norma_cuad = producto_discreto(y_datos, y_datos)
    correccion = sum(coefs[i] * producto_discreto(y_datos, base_vals[i]) for i in range(n))
    E = np.sqrt(abs(norma_cuad - correccion))
    print(f"\n  Error del método: E = {E:.10f}")

    print(f"\n  {'x':>6}   {'y real':>8}   {'f*(x)':>10}   {'dif':>10}")
    for xi, yi in zip(x_datos, y_datos):
        fi = f_aprox(xi)
        print(f"  {xi:>6.1f}   {yi:>8.5f}   {fi:>10.6f}   {abs(yi-fi):>10.6f}")

    print(f"\n  Predicción E = 2.5 pulg:  t = {f_aprox(2.5):.6f} min")

    x_plot = np.linspace(0.8, 5.2, 300)
    plt.figure(figsize=(9, 5))
    plt.scatter(x_datos, y_datos, color="blue", s=100, zorder=5, label="Datos reales")
    plt.plot(x_plot, [f_aprox(x) for x in x_plot], "r-", linewidth=2, label="f*(x) Chebyshev")
    plt.title(f"Ejemplo Discreta (Soplete)  —  Error = {E:.6f}")
    plt.xlabel("Espesor (pulgadas)"); plt.ylabel("Tiempo (min)")
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.tight_layout(); plt.show()


def ejemplo_caso1():
    """Ejemplo caso 1: f(x) = a*e^(b*x + c*x²)"""
    print("\n" + "=" * 60)
    print("  EJEMPLO — Caso 1: f(x) = a * e^(b*x + c*x²)")
    print("  Datos: temperatura de dispositivo vs tiempo de uso (min)")
    print("=" * 60)

    x_datos = [0, 5, 10, 15, 20, 25, 30]
    y_datos = [30, 34, 41, 51, 67, 92, 134]

    print("\n  Datos:")
    print(f"  {'Tiempo (min)':>14}   {'Temp (°C)':>10}")
    for xi, yi in zip(x_datos, y_datos):
        print(f"  {xi:>14.0f}   {yi:>10.1f}")

    p, a, b, c = resolver_caso1(x_datos, y_datos)
    E = mostrar_tabla(x_datos, y_datos, p)
    graficar(x_datos, y_datos, p,
             f"Ejemplo Caso 1 — Error = {E:.5f}",
             f"p(x) = {a:.3f}·e^({b:.4f}x + {c:.5f}x²)",
             xlabel="Tiempo (min)", ylabel="Temperatura (°C)")


def ejemplo_caso2():
    """Ejemplo caso 2: f(x) = a*cos(b*x + π),  b=1, a=4"""
    print("\n" + "=" * 60)
    print("  EJEMPLO — Caso 2: f(x) = a * cos(b*x + π)")
    print("  Datos generados con a=4, b=1  (b conocido)")
    print("=" * 60)

    x_datos = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    y_datos = [-4.00, -3.51, -2.16, -0.28, 1.66, 3.20, 3.96, 3.75, 2.61]
    b_val   = 1.0

    print("\n  Datos:")
    print(f"  {'x':>6}   {'y':>8}")
    for xi, yi in zip(x_datos, y_datos):
        print(f"  {xi:>6.2f}   {yi:>8.3f}")
    print(f"\n  b = {b_val}  (estimado del período: T = 2π ≈ 6.28)")

    p, a = resolver_caso2(x_datos, y_datos, b_val)
    E = mostrar_tabla(x_datos, y_datos, p)
    graficar(x_datos, y_datos, p,
             f"Ejemplo Caso 2 — Error = {E:.5f}",
             f"p(x) = {a:.4f}·cos({b_val}·x + π)")


def ejemplo_caso3():
    """Ejemplo caso 3: f(x) = a/(b+x),  a=6, b=2"""
    print("\n" + "=" * 60)
    print("  EJEMPLO — Caso 3: f(x) = a / (b + x)")
    print("  Datos generados con a=6, b=2")
    print("=" * 60)

    x_datos = [1, 2, 3, 4, 5, 6, 8, 10]
    y_datos = [2.000, 1.500, 1.200, 1.000, 0.857, 0.750, 0.600, 0.500]

    print("\n  Datos:")
    print(f"  {'x':>6}   {'y=6/(2+x)':>12}")
    for xi, yi in zip(x_datos, y_datos):
        print(f"  {xi:>6.0f}   {yi:>12.4f}")

    p, a, b = resolver_caso3(x_datos, y_datos)
    E = mostrar_tabla(x_datos, y_datos, p)
    graficar(x_datos, y_datos, p,
             f"Ejemplo Caso 3 — Error = {E:.8f}",
             f"p(x) = {a:.4f} / ({b:.4f} + x)")


def ejemplo_caso4():
    """Ejemplo caso 4: f(x) = a*x/(1+b*e^(c*x)),  a=10, b=5, c=-0.3"""
    print("\n" + "=" * 60)
    print("  EJEMPLO — Caso 4: f(x) = a*x / (1 + b*e^(c*x))")
    print("  Datos generados con a=10, b=5, c=-0.3  (c conocido)")
    print("=" * 60)

    x_datos = [1,    2,    4,     6,     8,     10,    12,    15,    20]
    y_datos  = [2.13, 5.34, 15.96, 32.84, 55.02, 80.06, 105.56, 142.07, 197.60]
    c_val   = -0.3

    print("\n  Datos:")
    print(f"  {'x':>6}   {'y':>10}")
    for xi, yi in zip(x_datos, y_datos):
        print(f"  {xi:>6.0f}   {yi:>10.3f}")
    print(f"\n  c = {c_val}  (conocido / estimado)")

    p, a, b = resolver_caso4(x_datos, y_datos, c_val)
    E = mostrar_tabla(x_datos, y_datos, p)
    graficar(x_datos, y_datos, p,
             f"Ejemplo Caso 4 — Error = {E:.5f}",
             f"p(x) = {a:.3f}·x / (1 + {b:.3f}·e^({c_val}·x))")


def ejemplo_ingenieria():
    """
    Aplicación a Ingeniería en Sistemas:
    Consumo de energía de un servidor en función de la carga del procesador.

    Problema: un servidor de datos tiene un sensor de consumo eléctrico.
    Se midió el consumo (Watts) para distintos niveles de carga de CPU (%).
    Se quiere modelar la relación para predecir el consumo en carga arbitraria
    y optimizar la gestión de energía del centro de datos.

    Modelo elegido: f(x) = a * e^(b*x + c*x²)
    Justificación: el consumo crece de forma super-lineal con la carga,
    acelerándose en cargas altas por el aumento de frecuencia y voltaje.
    """
    print("\n" + "=" * 60)
    print("  EJEMPLO DE INGENIERÍA EN SISTEMAS")
    print("  Consumo eléctrico de servidor vs carga de CPU")
    print("  Modelo: f(x) = a * e^(b*x + c*x²)")
    print("=" * 60)

    # Tabla de 11 puntos medidos en laboratorio
    x_datos = [0,  10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    y_datos  = [45, 50, 57, 66, 77, 91, 107, 126, 148, 174, 205]

    print("\n  Datos medidos:")
    print(f"  {'Carga CPU (%)':>15}   {'Consumo (W)':>12}")
    for xi, yi in zip(x_datos, y_datos):
        print(f"  {xi:>15.0f}   {yi:>12.1f}")

    print("\n  Aplicando modelo f(x) = a * e^(b*x + c*x²)...")
    p, a, b, c = resolver_caso1(x_datos, y_datos)

    E = mostrar_tabla(x_datos, y_datos, p)

    # Predicciones útiles para ingeniería
    print("\n  Predicciones para valores intermedios:")
    for xp in [25, 45, 55, 75, 85, 95]:
        print(f"    Carga {xp:>3}%  →  consumo estimado = {p(xp):.2f} W")

    # Gráfico con labels de ingeniería
    x_plot = np.linspace(0, 100, 300)
    y_plot = [p(x) for x in x_plot]

    plt.figure(figsize=(10, 5))
    plt.scatter(x_datos, y_datos, color="blue", s=100, zorder=5, label="Datos medidos")
    plt.plot(x_plot, y_plot, "r-", linewidth=2,
             label=f"p(x) = {a:.2f}·e^({b:.5f}x + {c:.6f}x²)")
    plt.title(f"Consumo de servidor vs carga CPU\nError = {E:.4f} W")
    plt.xlabel("Carga CPU (%)"); plt.ylabel("Consumo (Watts)")
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.tight_layout(); plt.show()


# ================================================================
# MENÚ PRINCIPAL
# ================================================================

def menu():
    while True:
        print("\n" + "=" * 60)
        print("  MÍNIMOS CUADRADOS — Análisis Numérico UM 2026")
        print("  Francisco Martin Gallardo | Santiago Ariel Martinez")
        print("=" * 60)
        print("  1. Aproximación discreta")
        print("  2. Aproximación discreta no lineal")
        print("  3. Ver ejemplos")
        print("  0. Salir")

        opcion = input("\n  Opción: ").strip()

        if opcion == "1":
            aproximacion_discreta()

        elif opcion == "2":
            aproximacion_no_lineal()

        elif opcion == "3":
            while True:
                print("\n" + "-" * 60)
                print("  EJEMPLOS")
                print("-" * 60)
                print("  1. Discreta         — soplete con Chebyshev")
                print("  2. No lineal caso 1 — f(x) = a·e^(b*x + c*x²)")
                print("  3. No lineal caso 2 — f(x) = a·cos(b*x + π)")
                print("  4. No lineal caso 3 — f(x) = a / (b + x)")
                print("  5. No lineal caso 4 — f(x) = a*x/(1 + b*e^(c*x))")
                print("  6. Ingeniería       — consumo de servidor vs carga CPU")
                print("  0. Volver")

                sub = input("\n  Ejemplo: ").strip()
                if sub == "1":   ejemplo_discreta()
                elif sub == "2": ejemplo_caso1()
                elif sub == "3": ejemplo_caso2()
                elif sub == "4": ejemplo_caso3()
                elif sub == "5": ejemplo_caso4()
                elif sub == "6": ejemplo_ingenieria()
                elif sub == "0": break
                else: print("  Opción no válida.")
                if sub != "0":
                    input("\n  [Enter para continuar...]")

        elif opcion == "0":
            print("\n  Chau!\n")
            break
        else:
            print("  Opción no válida.")

        if opcion != "3":
            input("\n  [Enter para continuar...]")


if __name__ == "__main__":
    menu()