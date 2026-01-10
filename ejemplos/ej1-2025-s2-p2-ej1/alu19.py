def calcular_promedio(nota1, nota2):
    promedio = []
    for m in range(len(nota1)):
        aux = (nota1[m] + nota2[m]) / 2
        promedio.append(aux)
    return promedio


def calcular_recursantes(promedio):
    qty = 0
    for m in range(len(promedio)):
        if promedio[m] <= 3:
            qty += 1
    return qty


def recursantes(nombre, apellidos, nota1, nota2, promedio):
    qty = calcular_recursantes(promedio)
    count = 0
    if qty:
        print("Almunos que recursan la materia:")
        for m in range(len(promedio)):
            if count % 3 == 0 and count:
                print("----------------------------")
            if promedio[m] <= 3:
                print(f"{nombre[m]} {apellidos[m]} - Promedio: {promedio[m]}")
                count += 1
    else:
        print("No hay recursantes")


def ordenar1(lista, lista2, lista3, lista4, lista5, lista6):
    for i in range(len(lista)):
        for j in range(i + 1, len(lista)):
            if lista[i] < lista[j]:
                lista[i], lista[j] = lista[j], lista[i]
                lista2[i], lista2[j] = lista2[j], lista2[i]
                lista3[i], lista3[j] = lista3[j], lista3[i]
                lista4[i], lista4[j] = lista4[j], lista4[i]
                lista5[i], lista5[j] = lista5[j], lista5[i]
                lista6[i], lista6[j] = lista6[j], lista6[i]


def largo_apellido(apellidos):
    ceros = []
    for m in range(len(apellidos)):
        ceros.append(len(apellidos[m]))
    return ceros


def apellidos_fun(nombre, apellidos, nota1, nota2, promedio):
    largo = largo_apellido(apellidos)
    ordenar1(largo, nombre, apellidos, promedio, nota1, nota2)
    ln = largo[0]
    print("notas de apellidos mas largos del segundo parcial")
    for m in range(len(largo)):
        if largo[m] == ln:
            print(f"{apellidos[m]} - Nota 2P: {nota2[m]}")


def orden2(nom, ape):
    for m in range(len(nom)):
        for j in range(len(nom) - 1 - m):
            if nom[m] > nom[m + 1]:
                nom[m], nom[m + 1] = nom[m + 1], nom[m]
                ape[m], ape[m + 1] = ape[m + 1], ape[m]


def promocion(nombre, apellidos, nota1, nota2, promedio):
    nom = []
    ape = []
    for m in range(len(nombre)):
        if promedio[m] > 7 and (6 <= nota1[m] < 7 or 6 <= nota2[m] < 7):
            nom.append(nombre[m])
            ape.append(apellidos[m])
    if ape:
        orden2(nom, ape)
        print("Almunos a revisar promocion")
        for m in range(len(ape)):
            print(f"{nom[m]} {ape[m]}")
    else:
        print("No hay alumnos a revisar promocion")


def mostrar_cursada(nombre, apellidos, nota1, nota2):
    if not nombre:
        print("No hay notas cargadas")
    else:
        promedio = calcular_promedio(nota1, nota2)
        recursantes(nombre, apellidos, nota1, nota2, promedio)
        print("\n")
        apellidos_fun(nombre, apellidos, nota1, nota2, promedio)
        print("\n")
        promocion(nombre, apellidos, nota1, nota2, promedio)
