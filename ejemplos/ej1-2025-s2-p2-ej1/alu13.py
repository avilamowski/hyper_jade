def mostrar_cursada(nombres, apellidos, notas1p, notas2p):
    if (
        len(nombres) == 0
        or len(apellidos) == 0
        or len(notas1p) == 0
        or len(notas2p) == 0
    ):
        print("No hay notas cargadas")
    else:
        print("\nAlumnos que recursan:")
        recursantes(nombres, apellidos, notas1p, notas2p)

        print("\nNotas del 2do parcial de apellidos más largos:")
        apellidos_largos(apellidos, notas2p)

        print("\nAlumnos con posibles promociones:")
        promocionados(nombres, apellidos, notas1p, notas2p)


def recursantes(nombres, apellidos, notas1p, notas2p):
    contador = 0
    i = 0
    while i < len(nombres):
        promedio = (notas1p[i] + notas2p[i]) / 2
        if promedio <= 3:
            print(nombres[i], apellidos[i], "- Promedio:", round(promedio, 2))
            contador = contador + 1
            if contador == 3:
                print("---------------------------------")
                contador = 0
        i = i + 1


def apellidos_largos(apellidos, notas2p):
    if len(apellidos) > 0:
        mayor = len(apellidos[0])
        i = 1
        while i < len(apellidos):
            if len(apellidos[i]) > mayor:
                mayor = len(apellidos[i])
            i = i + 1

        j = 0
        while j < len(apellidos):
            if len(apellidos[j]) == mayor:
                print(apellidos[j], "- Nota 2p:", notas2p[j])
            j = j + 1


def promocionados(nombres, apellidos, notas1p, notas2p):
    prom_nombres = []
    prom_apellidos = []

    i = 0
    while i < len(nombres):
        n1 = notas1p[i]
        n2 = notas2p[i]
        promedio = (notas1p[i] + notas2p[i]) / 2
        if (promedio > 7) and (6 <= n1 < 7 or 6 <= n2 < 7):
            prom_nombres = prom_nombres + [nombres[i]]
            prom_apellidos = prom_apellidos + [apellidos[i]]
        i = i + 1

    k = 0
    while k < len(prom_nombres) - 1:
        m = k + 1
        while m < len(prom_nombres):
            if prom_nombres[k] > prom_nombres[m]:
                aux_nombre = prom_nombres[k]
                aux_apellido = prom_apellidos[k]
                prom_nombres[k] = prom_nombres[m]
                prom_apellidos[k] = prom_apellidos[m]
                prom_nombres[m] = aux_nombre
                prom_apellidos[m] = aux_apellido
            m = m + 1
        k = k + 1

    i = 0
    while i < len(prom_nombres):
        print(prom_nombres[i], prom_apellidos[i])
        i = i + 1


mostrar_cursada(nombres, apellidos, notas1p, notas2p)
