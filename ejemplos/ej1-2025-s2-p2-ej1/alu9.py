def promediar(notas1p, notas2p):
    promedios = []
    for i in range(len(notas1p)):
        aux = (notas1p[i] + notas2p[i]) / 2
        aux = round((aux), 2)
        promedios.append(aux)
    return promedios


def recursar(nombres, apellidos, promedios):
    recursantes_nombres = []
    recursantes_apellidos = []
    promedio_recursantes = []
    for i in range(len(promedios)):
        if promedios[i] <= 3:
            recursantes_nombres.append(nombres[i])
            recursantes_apellidos.append(apellidos[i])
            promedio_recursantes.append(promedios[i])
    return recursantes_nombres, recursantes_apellidos, promedio_recursantes


def estructura_recursante(
    recursantes_nombres, recursantes_apellidos, promedio_recursantes
):
    print("Alumnos que recursan la materia:")
    for i in range(len(recursantes_nombres)):
        if i != 0 and (i / 3) == True:
            print("----------")
            print(
                f"{recursantes_nombres[i]} {recursantes_apellidos[i]} - Promedio: {promedio_recursantes[i]}"
            )
        else:
            print(
                f"{recursantes_nombres[i]} {recursantes_apellidos[i]} - Promedio: {promedio_recursantes[i]}"
            )


def apellido_largo_lista(apellidos, notas2p):
    apellidos_largos = []
    notas2p_apellidos_largos = []
    aux = 0
    for i in range(len(apellidos)):
        if len(apellidos[i]) > aux:
            aux = len(apellidos[i])
    for i in range(len(apellidos)):
        if len(apellidos[i]) == aux:
            apellidos_largos.append(apellidos[i])
            notas2p_apellidos_largos.append(notas2p[i])
    return notas2p_apellidos_largos, apellidos_largos


def estructura_apellidos_largos(notas2p_apellidos_largos, apellidos_largos):
    print("\nNotas del 2do parcial de apellidos más largos")
    for i in range(len(apellidos_largos)):
        print(f"{apellidos_largos[i]} - Nota 2P: {notas2p_apellidos_largos[i]}")


def promocionar(nombres, apellidos, promedios, notas1p, notas2p):
    posible_prom_nombre = []
    posible_prom_apellido = []
    for i in range(len(notas1p)):
        if promedios[i] > 7 and (
            (notas1p[i] >= 6 and notas1p[i] < 7) or (notas2p[i] >= 6 and notas2p[i] < 7)
        ):
            posible_prom_nombre.append(nombres[i])
            posible_prom_apellido.append(apellidos[i])
    return posible_prom_nombre, posible_prom_apellido


def ordenar_alfabeticamente(posible_prom_nombre, posible_prom_apellido):
    for i in range(len(posible_prom_nombre)):
        for j in range(i + 1, len(posible_prom_nombre)):
            if posible_prom_nombre[i] < posible_prom_nombre[j]:
                posible_prom_nombre[i], posible_prom_nombre[j] = (
                    posible_prom_nombre[j],
                    posible_prom_nombre[i],
                )
                posible_prom_apellido[i], posible_prom_apellido[j] = (
                    posible_prom_apellido[j],
                    posible_prom_apellido[i],
                )
            if posible_prom_nombre[i] == posible_prom_nombre[j]:
                if posible_prom_apellido[i] < posible_prom_apellido[j]:
                    posible_prom_nombre[i], posible_prom_nombre[j] = (
                        posible_prom_nombre[j],
                        posible_prom_nombre[i],
                    )
                    posible_prom_apellido[i], posible_prom_apellido[j] = (
                        posible_prom_apellido[j],
                        posible_prom_apellido[i],
                    )
    return posible_prom_nombre, posible_prom_apellido


def estructura_promocionar(posible_prom_nombre, posible_prom_apellido):
    print("\nAlumnos que se debe revisar posible promoción:")
    for i in range(len(posible_prom_nombre)):
        print(f"{posible_prom_nombre[i]} {posible_prom_apellido[i]}")


nombres = [
    "Ana",
    "Juan",
    "Luis",
    "María",
    "Lucía",
    "Ruben",
    "Adrian",
    "Jorge",
    "Adrian",
    "Zenon",
]
apellidos = [
    "Li",
    "Gómez",
    "Paz",
    "Sosa",
    "Ro",
    "Paz",
    "Martinez",
    "Carranza",
    "Casio",
    "Perok",
]
notas1p = [1, 2.0, 0, 8, 6.50, 6.5, 6.5, 1, 6, 6.7]
notas2p = [1, 1.57, 3, 7, 9, 9, 10, 5, 10, 9]


def mostrar_cursada(nombres, apellidos, notas1p, notas2p):
    if not len(nombres) == 0:
        promedios = promediar(notas1p, notas2p)
        recursantes_nombres, recursantes_apellidos, promedio_recursantes = recursar(
            nombres, apellidos, promedios
        )
        estructura_recursante(
            recursantes_nombres, recursantes_apellidos, promedio_recursantes
        )

        notas2p_apellidos_largos, apellidos_largos = apellido_largo_lista(
            apellidos, notas2p
        )
        estructura_apellidos_largos(notas2p_apellidos_largos, apellidos_largos)

        posible_prom_nombre, posible_prom_apellido = promocionar(
            nombres, apellidos, promedios, notas1p, notas2p
        )
        posible_prom_nombre, posible_prom_apellido = ordenar_alfabeticamente(
            posible_prom_nombre, posible_prom_apellido
        )
        estructura_promocionar(posible_prom_nombre, posible_prom_apellido)
    else:
        print("No hay notas cargadas")


mostrar_cursada(nombres, apellidos, notas1p, notas2p)
