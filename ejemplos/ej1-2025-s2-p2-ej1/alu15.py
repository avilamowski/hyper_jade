nombres = []
apellidos = []
notas1p = []
notas2p = []


def mostrar_cursada(nombres, apellidos, notas1p, notas2p):
    if not (
        len(nombres) == 0
        and len(apellidos) == 0
        and len(notas1p) == 0
        and len(notas2p) == 0
    ):
        contador = 0
        promedios = []
        print("los que recursan son: ")
        for c in range(len(notas1p)):
            promedio = 0
            promedio = round(((notas1p[c] + notas2p[c]) / 2), 2)
            promedios.append(promedio)
            if promedio < 4:
                if contador == 0:
                    print(f"{nombres[c]} {apellidos[c]}")
                    contador += 1
                elif not (contador % 3 == 0):
                    print(f"{nombres[c]} {apellidos[c]}")
                    contador += 1
                else:
                    print("-----------")
                    print(f"{nombres[c]} {apellidos[c]}")

        print("")
        for i in range(len(apellidos)):
            for j in range(i + 1, len(apellidos)):
                if len(apellidos[i]) < len(apellidos[j]):
                    apellidos[i], apellidos[j] = apellidos[j], apellidos[i]
                    nombres[i], nombres[j] = nombres[j], nombres[i]
                    notas1p[i], notas1p[j] = notas1p[j], notas1p[i]
                    notas2p[i], notas2p[j] = notas2p[j], notas2p[i]
                    promedios[i], promedios[j] = promedios[j], promedios[i]
        apellido_largo = []
        print("Notas del 2do parcial de apellidos más largos")
        for c in range(len(apellidos)):
            if len(apellidos[0]) == len(apellidos[c]):
                apellido_largo.append(apellidos[c])
                print(f"{apellidos[c]} con nota del segundo parcial {notas2p[c]}")
        print("")
        print("los alumnos que puede promocionar son")
        for k in range(len(promedios)):
            if promedios[k] > 7:
                if (notas1p[k] < 7 and notas2p[k] >= 6) or (
                    notas1p[k] >= 6 and notas2p[k] < 7
                ):
                    print(f"{nombres[k]} {apellidos[k]}")

    else:
        print("No hay notas cargadas")


mostrar_cursada(nombres, apellidos, notas1p, notas2p)
