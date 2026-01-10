def prom(notas1, notas2):
    recu = []
    for i in range(len(notas1)):
        prom = round((notas1[i] + notas2[i]) / 2, 2)
        recu.append(prom)
        # print(prom)
    return recu


def recursar(nombres, apellidos, notas1, notas2):
    promedio = prom(notas1, notas2)
    recu = []
    for i in range(len(promedio)):
        if promedio[i] <= 3:
            recu.append(i)
    print("alumnos que recursan la materia: ")
    cont = 0
    for i in recu:
        cont += 1
        if cont <= 3:
            print("\n")
            print(f"{nombres[i]} {apellidos[i]} - promedio: {promedio[i]}")
        else:
            print("-------------")
            print(f"{nombres[i]} {apellidos[i]} - promedio: {promedio[i]}")


def ordenar(apellidos, notas2):
    apellidosf = apellidos[:]
    notas2f = notas2[:]
    for i in range(len(apellidosf)):
        for j in range(i + 1, len(apellidosf)):
            if len(apellidosf[i]) < len(apellidosf[j]):
                apellidosf[i], apellidosf[j] = apellidosf[j], apellidosf[i]
                notas2f[i], notas2f[j] = notas2f[j], notas2f[i]
    print("\n")
    print("Notas del 2do parcial de apellidos más largos: ")
    len_max = len(apellidos[0])
    for i in range(len(apellidos)):
        if len(apellidos[i]) == len_max:
            print(f"{apellidos[i]} - nota 2P: {notas2f[i]}")


def ordenar2(posibles_n, posibles_a):
    for i in range(len(posibles_n)):
        for j in range(i + 1, len(posibles_n)):
            if (
                posibles_n[i] < posibles_n[j]
            ):  ##asumos que el nombre siempre tiene mayusculas ya que asi esta en todos los ejemplos
                posibles_n[i], posibles_n[j] = posibles_n[j], posibles_n[i]
                posibles_a[i], posibles_a[j] = posibles_a[j], posibles_a[i]
    print("\n")
    print("alumnos que se debe revisar la promocion: ")
    for i in range(len(posibles_n)):
        print(f"{posibles_n[i]} {posibles_a[i]}")


def promocion(nombres, apellidos, notas1, notas2):
    promedio = prom(notas1, notas2)
    posibles_n = []
    posibles_a = []
    print(promedio)
    for i in range(len(notas1)):
        if promedio[i] > 7:
            # print(notas1[i], notas2[i])
            if (notas1[i] >= 6 and notas1[i] < 7) or (notas2[i] < 7 and notas2[i] >= 6):
                posibles_n.append(nombres[i])
                posibles_a.append(apellidos[i])
    ordenar2(posibles_n, posibles_a)


def mostrar_cursada(nombres, apellidos, notas1, notas2):
    recursar(nombres, apellidos, notas1, notas2)
    ordenar(apellidos, notas2)
    promocion(nombres, apellidos, notas1, notas2)


nombres = ["Ana", "Juan", "Luis", "María", "Lucía", "Ruben", "Adrian", "Jorge"]
apellidos = ["Li", "Gómez", "Paz", "Sosa", "Ro", "Paz", "Martinez", "Carranza"]
notas1 = [1, 2, 0, 8, 6.50, 9, 9, 1]
notas2 = [1, 1.50, 3, 7, 9, 7, 10, 5]


if len(nombres) == 0 and len(apellidos) == 0 and len(notas1) == 0 and len(notas2) == 0:
    print("No hay notas cargadas")
else:
    mostrar_cursada(nombres, apellidos, notas1, notas2)
