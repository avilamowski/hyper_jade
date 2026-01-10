def mostrar_cursada(nombres, apellidos, notas1p, notas2p):
    if len(nombres) == 0:
        print("No hay notas cargadas.")
        return

    # Primera parte, los promedios
    print("Alumnos que recursan la materia: \n")
    contador = 0
    for i in range(len(nombres)):
        promedio = (notas1p[i] + notas2p[i]) / 2

        if promedio <= 3:
            print(f"-{nombres[i]} {apellidos[i]} - Promedio: {round(promedio, 2)}")
            contador += 1

            if contador % 3 == 0:
                print("--------------------------------")

    # Segunda parte, apeliidos mas largos
    print("\n")
    print("Notas del segundo parcial de apellidos mas largos: \n")
    max_s = len(apellidos[0])

    for i in apellidos:
        if len(i) > max_s:
            max_s = len(i)

    for i in range(len(apellidos)):
        if len(apellidos[i]) == max_s:
            print(f"-{apellidos[i]} - Nota 2P: {notas2p[i]}")

    # Tercera parte, posibles promociones
    print("\n")
    print("Alumnos que deben verificar posible promocion: \n")
    promocion = []

    for i in range(len(nombres)):
        promedio = (notas1p[i] + notas2p[i]) / 2

        if promedio > 7 and (6 <= notas1p[i] < 7 or 6 <= notas2p[i] < 7):
            promocion.append(f"{nombres[i]} {apellidos[i]}")

    # Ordenar la lista de forma descendiente:
    for i in range(len(promocion)):
        for j in range(i + 1, len(promocion)):
            if promocion[i] < promocion[j]:
                promocion[i], promocion[j] = promocion[j], promocion[i]

    for i in promocion:
        print(i)


nombres = [
    "Ana",
    "Juan",
    "Luis",
    "María",
    "Lucía",
    "Ruben",
    "Adrian",
    "Jorge",
    "Agustin",
    "Bautista",
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
    "Alvarez",
    "Balza",
]
notas1p = [1, 2, 0, 8, 6.50, 9, 9, 1, 6, 6.5]
notas2p = [1, 1.50, 3, 7, 9, 7, 10, 5, 10, 9]

mostrar_cursada(nombres, apellidos, notas1p, notas2p)
