def ordenar_custom(lista):
    for i in range(len(lista)):
        for j in range(i + 1, len(lista)):
            if lista[i] > lista[j]:
                lista[i], lista[j] = lista[j], lista[i]
    return lista


def recursan(nombres, apellidos, notas1p, notas2p):
    cant_prom = 0
    for i in range(len(nombres)):
        if ((notas1p[i] + notas2p[i]) / 2) <= 3:
            print(
                f"{nombres[i]} {apellidos[i]} - Promedio: {round(((notas1p[i] + notas2p[i]) / 2), 2)}"
            )
            cant_prom += 1
            if cant_prom % 3 == 0:
                print("----------------------")


def apellidos_largos(nombres, apellidos, notas2p):
    max_len = 0
    apellidos_largos = []
    notas_apellidos_largos = []
    for i in range(len(apellidos)):
        if len(apellidos[i]) > max_len:
            max_len = len(apellidos[i])
    for i in range(len(apellidos)):
        if len(apellidos[i]) == max_len:
            apellidos_largos += [apellidos[i]]
            notas_apellidos_largos += [notas2p[i]]
    for i in range(len(apellidos_largos)):
        print(f"{apellidos_largos[i]} - Nota 2P: {notas_apellidos_largos[i]}")


def posible_promocion(nombres, apellidos, notas1p, notas2p):
    revision = []
    for i in range(len(nombres)):
        if ((notas1p[i] + notas2p[i]) / 2) >= 7:
            if 7 > notas1p[i] >= 6 and notas2p[i] >= 7:
                revision += [f"{nombres[i]} {apellidos[i]}"]
            if 7 > notas2p[i] >= 6 and notas1p[i] >= 7:
                revision += [f"{nombres[i]} {apellidos[i]}"]
    revision_ordenada = ordenar_custom(revision)
    for alumno in revision_ordenada:
        print(alumno)


def mostrar_cursada(nombres, apellidos, notas1p, notas2p):
    if (
        len(nombres) == 0
        or len(apellidos) == 0
        or len(notas1p) == 0
        or len(notas2p) == 0
    ):
        print("No hay notas cargadas")
    else:
        print("Alumnos que recursan la materia: ")
        recursan(nombres, apellidos, notas1p, notas2p)
        print("\nNotas del 2do parcial de apellidos más largos: ")
        apellidos_largos(nombres, apellidos, notas2p)
        print("\nAlumnos que se debe revisar posible promoción: ")
        posible_promocion(nombres, apellidos, notas1p, notas2p)


def main():
    nombres = ["Ana", "Juan", "Luis", "María", "Lucía", "Ruben", "Adrian", "Jorge"]
    apellidos = ["Li", "Gómez", "Paz", "Sosa", "Ro", "Paz", "Martinez", "Carranza"]
    notas1p = [1, 2, 0, 8, 6.50, 9, 9, 1]
    notas2p = [1, 1.50, 3, 7, 9, 7, 10, 5]
    mostrar_cursada(nombres, apellidos, notas1p, notas2p)


main()
