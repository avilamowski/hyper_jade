def recursados(nombres, apellidos, notas1p, notas2p, prom):
    cont = 0
    print("Alumnos que recursaron la materia:")
    for i in range(len(prom)):
        if prom[i] <= 3:
            cont += 1
            if cont <= 3:
                print(f"{nombres[i]} {apellidos[i]} - promedio:{round(prom[i], 2)}")
            elif cont > 3:
                print("----------------------------")
                print(f"{nombres[i]} {apellidos[i]} - promedio:{round(prom[i], 2)}")
        suma = 0
    if cont == 0:
        print("No hubo alumnos que recursaron")


def promedios(notas1p, notas2p):
    prom = []
    for i in range(len(notas1p)):
        suma = 0
        suma += notas1p[i]
        suma += notas2p[i]
        p = suma / 2
        prom.append(p)
    return prom


def apellido_mas_largo(apellidos, notas2p):
    print("Notas del 2do parcial de apellidos más largos")
    ap_largo = ""
    largo = 0
    for i in range(len(apellidos)):
        if len(apellidos[i]) > largo:
            ap_largo = apellidos[i]
            largo = len(apellidos[i])

    for i in range(len(apellidos)):
        if len(apellidos[i]) == largo:
            print(f"{apellidos[i]} - Nota 2P: {notas2p[i]}")


def posible_promocion(prom, nombres, apellidos, notas1p, notas2p):
    print("Alumnos que se debe revisar posible promoción:")
    cont = 0
    for i in range(len(prom)):
        if prom[i] > 7:
            if (notas1p[i] < 7 and notas1p[i] >= 6) or (
                notas2p[i] < 7 and notas2p[i] >= 6
            ):
                cont += 1
                if cont > 0:
                    print(f"{nombres[i]} {apellidos[i]}")
    if cont == 0:
        print("No  hay alumnos que se deba revisar su promoción")


nombres = ["Ana", "Juan", "Luis", "María", "Lucía", "Ruben", "Adrian", "Jorge"]
apellidos = ["Li", "Gómez", "Paz", "Sosa", "Ro", "Paz", "Martinez", "Carranza"]
notas1p = [1, 2, 0, 8, 6.50, 9, 9, 1]
notas2p = [1, 1.50, 3, 7, 9, 7, 10, 5]


def mostrar_cursada(nombres, apellidos, notas1p, notas2p):
    if nombres != [] and apellidos != [] and notas1p != [] and notas2p != []:
        prom = promedios(notas1p, notas2p)
        recursados(nombres, apellidos, notas1p, notas2p, prom)
        apellido_mas_largo(apellidos, notas2p)
        posible_promocion(prom, nombres, apellidos, notas1p, notas2p)
    else:
        print("No hay notas cargadas")


mostrar_cursada(nombres, apellidos, notas1p, notas2p)
