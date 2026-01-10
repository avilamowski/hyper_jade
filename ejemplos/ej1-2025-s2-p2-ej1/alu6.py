nombres = ["Ana", "Juan", "Luis", "María", "Lucía", "Ruben", "Adrian", "Jorge"]
apellidos = ["Li", "Gómez", "Paz", "Sosa", "Ro", "Paz", "Martinez", "Carranza"]
notas1p = [1, 2, 0, 8, 6.50, 9, 9, 1]
notas2p = [1, 1.50, 3, 7, 9, 7, 10, 5]
# nombres = []
# apellidos = []
# notas1p = []
# notas2p = []


def promedio(nombres, apellidos, notas1p, notas2p):
    contador = 0
    print("Alumnos que recursan la materia: \n")
    for i in range(len(nombres)):
        promedio = round((notas1p[i] + notas2p[i]) / 2, 2)
        if promedio <= 3:
            if contador % 3 == 0 and contador != 0:
                print("\n ----------------- \n")
            print(f"\n{nombres[i]} {apellidos[i]} - Promedio {promedio}: \n")
            contador += 1

    return nombres, apellidos, notas1p, notas2p


# promedio(nombres, apellidos, notas1p, notas2p)


def ordenar_custom(nombres, apellidos, notas1p, notas2p):
    for i in range(len(apellidos)):
        for j in range(i + 1, len(apellidos)):
            if len(apellidos[i]) < len(apellidos[j]):
                apellidos[j], apellidos[i] = apellidos[i], apellidos[j]
                nombres[j], nombres[i] = nombres[i], nombres[j]
                notas1p[j], notas1p[i] = notas1p[i], notas1p[j]
                notas2p[j], notas2p[i] = notas2p[i], notas2p[j]

    return nombres, apellidos, notas1p, notas2p


# print(ordenar_custom(nombres, apellidos, notas1p, notas2p))


def notas_largos(nombres, apellidos, notas1p, notas2p):
    nombres, apellidos, notas1p, notas2p = ordenar_custom(
        nombres, apellidos, notas1p, notas2p
    )
    apellido_largo = apellidos[0]
    print("Notas del 2do parcial de apellidos más largos\n")
    for i in range(len(apellidos)):
        if len(apellido_largo) == len(apellidos[i]):
            print(f"\n{apellidos[i]} - Nota 2P: {notas2p[i]}\n")

    return nombres, apellidos, notas1p, notas2p


# notas_largos(nombres, apellidos, notas1p, notas2p)


def ordenarlos_alfabeticamente(nombres, apellidos, notas1p, notas2p):
    for i in range(len(apellidos)):
        for j in range(i + 1, len(apellidos)):
            if nombres[i] < nombres[j]:
                apellidos[i], apellidos[j] = apellidos[j], apellidos[i]
                nombres[i], nombres[j] = nombres[j], nombres[i]
                notas1p[i], notas1p[j] = notas1p[j], notas1p[i]
                notas2p[i], notas2p[j] = notas2p[j], notas2p[i]

    return nombres, apellidos, notas1p, notas2p


def posible_promocion(nombres, apellidos, notas1p, notas2):
    nombres, apellidos, notas1p, notas2p = ordenarlos_alfabeticamente(
        nombres, apellidos, notas1p, notas2
    )
    print("\nAlumnos que se debe revisar posible promoción:\n")
    for i in range(len(apellidos)):
        promedio = (notas1p[i] + notas2p[i]) / 2
        if promedio > 7 and (7 > notas1p[i] >= 6 or 7 > notas2p[i] >= 6):
            print(f"{nombres[i]} {apellidos[i]}")
    return nombres, apellidos, notas1p, notas2p


# posible_promocion(nombres, apellidos, notas1p, notas2p)


def mostrar_cursada(nombres, apellidos, notas1p, notas2):
    if nombres == [] and apellidos == [] and notas1p == [] and notas2p == []:
        print("No hay notas cargadas")

    else:
        promedio(nombres, apellidos, notas1p, notas2p)
        notas_largos(nombres, apellidos, notas1p, notas2p)
        posible_promocion(nombres, apellidos, notas1p, notas2)
        ordenar_custom(nombres, apellidos, notas1p, notas2p)

    return mostrar_cursada


mostrar_cursada(nombres, apellidos, notas1p, notas2p)
# mostrar_cursada([], [], [], [])
