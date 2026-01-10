nombres = ["Ana", "Juan", "Luis", "María", "Lucía", "Ruben", "Adrian", "Jorge"]
apellidos = ["Li", "Gómez", "Paz", "Sosa", "Ro", "Paz", "Martinez", "Carranza"]
notas1p = [1, 2, 0, 8, 6.50, 9, 9, 1]
notas2p = [1, 1.50, 3, 7, 9, 7, 10, 5]


# ACA COMIENZA MI CODIGO
def promedio(notas1p, notas2p):
    promedios = []
    for nota in range(len(notas1p)):
        promedio = (notas1p[nota] + notas2p[nota]) / 2
        promedios.append(promedio)
    return promedios


def recursar(nombres, apellidos, notas1p, notas2p):
    promedios = promedio(notas1p, notas2p)
    print("Alumnos que recursan la materia: ")
    contador = 0
    for i in range(len(nombres)):
        if promedios[i] <= 3:
            print(nombres[i], apellidos[i], "- Promedio: ", promedios[i])
            contador += 1
            if contador % 3 == 0:
                print("--------------------")


def mas_largo(apellidos, notas2p):
    print("\nNotas del segundo parcial de apellidos mas largos: ")
    max = len(apellidos[0])
    for i in range(len(apellidos)):
        if len(apellidos[i]) >= max:
            max = len(apellidos[i])
    for i in range(len(apellidos)):
        if len(apellidos[i]) == max:
            print(f"{apellidos[i]} - Nota 2P: {notas2p[i]}")


def ordenar_alf(nombres, apellidos, notas1p, notas2p):
    promedios = promedio(notas1p, notas2p)
    for i in range(len(nombres)):
        for j in range(i + 1, len(nombres)):
            if nombres[i] > nombres[j]:
                nombres[i], nombres[j] = nombres[j], nombres[i]
                apellidos[i], apellidos[j] = apellidos[j], apellidos[i]
                notas1p[i], notas1p[j] = notas1p[j], notas1p[i]
                notas2p[i], notas2p[j] = notas2p[j], notas2p[i]
                # promedios[i], promedios[j] = promedios[j], promedios[i]
    return nombres, apellidos, notas1p, notas2p


def promocion(nombres, apellidos, notas1p, notas2p):
    nombres, apellidos, notas1p, notas2p = ordenar_alf(
        nombres, apellidos, notas1p, notas2p
    )
    print("\nAlumnos que se debe revisar posible promocion: ")
    for i in range(len(nombres)):
        p1, p2 = notas1p[i], notas2p[i]
        # promedio = promedios[i]
        # cond1 = (promedio > 7)
        cond2 = p1 < 7 and p1 >= 6
        cond3 = p2 < 7 and p2 >= 6
        if cond2 or cond3:
            print(nombres[i] + " " + apellidos[i])
    return nombres, apellidos, notas1p, notas2p


# FUNCION FINAL
def mostrar_cursada(nombres, apellidos, notas1p, notas2p):
    if nombres == []:
        print("No hay notas cargadas")
        return
    else:
        recursar(nombres, apellidos, notas1p, notas2p)
        mas_largo(apellidos, notas2p)
        promocion(nombres, apellidos, notas1p, notas2p)


mostrar_cursada(nombres, apellidos, notas1p, notas2p)
