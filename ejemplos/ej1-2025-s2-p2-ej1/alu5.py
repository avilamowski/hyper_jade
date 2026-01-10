NOMBRES = ["Ana", "Juan", "Luis", "María", "Lucía", "Ruben", "Adrian", "Jorge"]
APELLIDOS = ["Li", "Gómez", "Paz", "Sosa", "Ro", "Paz", "Martinez", "Carranza"]
NOTAS1P = [1, 2, 0, 8, 6.50, 9, 9, 1]
NOTAS2P = [1, 1.50, 3, 7, 9, 7, 10, 5]


def apellidos_largos(lista, lista2, lista3, lista4):
    for i in range(len(lista)):
        for j in range(i + 1, len(lista)):
            if len(lista2[i]) < len(lista2[j]):
                lista[i], lista[j] = lista[j], lista[i]
                lista2[i], lista2[j] = lista2[j], lista2[i]
                lista3[i], lista3[j] = lista3[j], lista3[i]
                lista4[i], lista4[j] = lista4[j], lista4[i]
    return lista2


def posible_promocion(lista, lista2, lista3, lista4):
    nombres_posible = []
    apellidos_posible = []
    promedio_posible = []
    for i in range(len(lista)):
        n1 = lista3[i]
        n2 = lista4[i]
        prom = (n1 + n2) / 2
        if (prom > 7 and 7 > n1 >= 6) or (prom > 7 and 7 > n2 >= 6):
            nombres_posible.append(lista[i])
            apellidos_posible.append(lista2[i])
            promedio_posible.append(prom)
    return nombres_posible, apellidos_posible, promedio_posible


def mostrar_cursada(nombres, apellidos, notas1p, notas2p):
    try:
        if nombres != [] and apellidos != [] and notas1p != [] and notas2p != []:
            ### 1 ###
            nombres_recursan = []
            apellidos_recursan = []
            prom_recursan = []

            for i in range(len(nombres)):
                n1 = notas1p[i]
                n2 = notas2p[i]
                prom = (n1 + n2) / 2
                if prom <= 3:
                    nombres_recursan.append(nombres[i])
                    apellidos_recursan.append(apellidos[i])
                    prom_recursan.append(prom)

            # imprimo #

            u = 0
            print("Alumnos que recursan la materia:")
            for j in range(len(nombres_recursan)):
                print(
                    f"{nombres_recursan[j]} {apellidos_recursan[j]} - Promedio: {prom_recursan[j]}"
                )
                u += 1
                if u % 3 == 0:
                    print("---------------------")

            ### 2 ###

            apellidos_largos(nombres, apellidos, notas1p, notas2p)
            ###
            nombres_l = []
            apellidos_l = []
            notas1p_l = []
            notas2p_l = []
            ###
            nombres_l.append(nombres[0])
            apellidos_l.append(apellidos[0])
            notas1p_l.append(notas1p[0])
            notas2p_l.append(notas2p[0])
            ###
            for x in range(1, len(nombres)):
                if len(apellidos_l[0]) == len(apellidos[x]):
                    nombres_l.append(nombres[x])
                    apellidos_l.append(apellidos[x])
                    notas1p_l.append(notas1p[x])
                    notas2p_l.append(notas2p[x])

            ### imprimo ###
            print("\nNotas del 2do parcial de apellidos más largos:")
            for h in range(len(nombres_l)):
                print(f"{apellidos_l[h]} - Nota2P: {notas2p_l[h]}")

            ### 3 ###
            nombres_posible = []
            apellidos_posible = []
            promedio_posible = []

            print("\nAlumnos que se debe revisar posible promoción:")
            np, ap, pp = posible_promocion(nombres, apellidos, notas1p, notas2p)
            for j in range(len(np)):
                print(f"{np[j]} {ap[j]}")

        else:
            print("No hay notas cargadas")
    except IndexError or NameError or ValueError:
        print("Ocurrio un error")


mostrar_cursada(NOMBRES, APELLIDOS, NOTAS1P, NOTAS2P)
