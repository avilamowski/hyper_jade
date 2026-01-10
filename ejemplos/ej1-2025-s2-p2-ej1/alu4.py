# 1
def promedio(a, b):
    prom = (a + b) / 2

    return prom


def promedio_listas(notas1p, notas2p):
    promedio = []

    i = 0
    while i < len(notas1p):
        promedio.append((float(notas1p[i]) + float(notas2p[i])) / 2)

        i += 1
    return promedio


def nombres_recursa(nombres, apellidos, promedio):
    recursante = []
    i = 0
    while i < len(promedio):
        if promedio[i] <= 3:
            recursante.append(f"{nombres[i]}")
            recursante.append(f"{apellidos[i]}")
            recursante.append(f"{promedio[i]}")
        i += 1
    print(recursante)
    return recursante


def mostrar_recursante(recursantes, promedio):
    print("Alumnos que recursan la materia: ")

    contador = 0
    i = 0
    while i < len(recursantes):
        if i == 0:
            print(
                f"{recursantes[i]} {recursantes[i + 1]} - Promedio: {recursantes[i + 2]}"
            )
            contador += 1
        else:
            print(
                f"{recursantes[i + 2]} {recursantes[i + 3]} - Promedio: {recursantes[i + 4]}"
            )
            contador += 1
            if contador % 3 == 0:
                print("------------------------------")
                print(
                    f"{recursantes[i + 2]} {recursantes[i + 3]} - Promedio: {recursantes[i + 4]}"
                )
        i += 1


# 2
def ordenar(apellidos, notas2p):
    for i in range(apellidos):  # Recorre hasta el último
        for j in range(i + 1, apellidos):
            if len(apellidos[j]) > len(apellidos[i]):
                apellido[j], apellidos[i] = (
                    apellidos[i],
                    apellidos[j],
                )  # Intercambio de posiciones
                notas2p[j], notas2p[i] = notas2p[i], notas2p[j]

    return apellidos, notas2p


def mostrar_notas(apellidos, notas2p):
    apellido, nota = ordenar(apellidos, notas2p)
    print("Notas del 2do parcial de apellidos más largos")

    for i in apellido:
        print(f"apellido[i] - Nota 2p: {nota[i]}")


# 3
def revisar_prom(promedio, notas1p, notas2p, nombres, apellidos):
    nombres_revisar = []
    promedio = promedio_listas(notas1p, notas2p)
    for i in range(len(promedio)):
        if float(promedio[i]) >= 7 and (
            6 <= float(notas1p[i]) < 7 or 6 <= float(notas2p[i]) < 7
        ):
            nombres_revisar.append(f"{nombres[i]} {apellidos[i]}")

    return nombres


print(apellido_y_notas(apellidos, notas2p))


def mostrar_cursada(nombres, apellidos, notas1p, notas2p):
    promedio = promedio_listas(notas1p, notas2p)
    recursante = nombres_recursa(nombres, apellidos, promedio)
    print(mostrar_recursante(recursante, promedio))

    apellido, nota = ordenar(apellidos, notas2p)
    print(mostrar_notas(apellido, nota))


# main
if nombres == [] or apellidos == [] or notas1p == [] or notas2p == []:
    print("No hay notas cargadas")
else:
    mostrar_cursada(nombres, apellidos, notas1p, notas2p)
