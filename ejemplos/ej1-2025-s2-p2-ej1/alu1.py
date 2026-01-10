def calcular_promedio(lista1, lista2):
    lista_promedio = []
    for i in range(len(lista1)):
        promedio = (lista1[i] + lista2[i]) / 2
        lista_promedio.append(promedio)
    return lista_promedio


def apellido_mas_largo(lista_apellido, notas_segundo_parcial):
    apellidos_mas_largos = [""]
    nota_de_ellos = [""]
    for i in range(len(lista_apellido)):
        if len(lista_apellido[i]) > len(apellidos_mas_largos[0]):
            apellidos_mas_largos[0] = lista_apellido[i]
            nota_de_ellos[0] = notas_segundo_parcial[i]

    for k in range(len(lista_apellido)):
        if len(apellidos_mas_largos[0]) == len(lista_apellido[k]):
            if lista_apellido[k] != apellidos_mas_largos[0]:
                apellidos_mas_largos.append(lista_apellido[k])
                nota_de_ellos.append(notas_segundo_parcial[k])

    return apellidos_mas_largos, nota_de_ellos


def nombre_completo(nombre, apellido):
    nombre_completo = []
    for i in range(len(nombre)):
        nombre_completo.append(nombre[i] + " " + apellido[i])

    return nombre_completo


def posible_promocion(primero, segundo, l_promedio, l_alumno):
    alumnos_posibles_promocion = []
    for i in range(len(l_alumno)):
        if l_promedio[i] > 7:
            if (6 <= primero[i] < 7) or (6 <= segundo[i] < 7):
                alumnos_posibles_promocion.append(l_alumno[i])
    alumnos_posibles_promocion = ordenar_alfabeticamente(alumnos_posibles_promocion)
    return alumnos_posibles_promocion


def ordenar_alfabeticamente(lista1):
    for i in range(len(lista1) - 1):
        if lista1[i] > lista1[i + 1]:
            lista1[i], lista1[i + 1] = lista1[i + 1], lista1[i]
    return lista1


def recursados(promedio, nombre_apellido):
    contador = 0
    print("Alumnos que recursan la materia: ")
    for i in range(len(promedio)):
        if promedio[i] <= 3:
            print(f"{nombre_apellido[i]} - Promedio: {promedio[i]}")
            contador += 1
            if contador % 3 == 0 and i != 0:
                print("--------------------------")


def apellidos_impresion(apellido, nota):
    print("Notas del 2do parcial de apellidos más largos")
    for i in range(len(apellido)):
        print(f"{apellido[i]} - Nota2p : {nota[i]}")


def promocion_impresion(lista_posibles):
    print("Alumnos que se debe revisar posible promoción:")
    for i in range(len(lista_posibles)):
        print(lista_posibles[i])


def mostrar_cursada(nombres, apellidos, notas1p, notas2p):
    try:
        promedio = calcular_promedio(notas1p, notas2p)
        apellido_largo, su_nota = apellido_mas_largo(apellidos, notas2p)
        nombres_completos = nombre_completo(nombres, apellidos)
        alumnos_posibles = posible_promocion(
            notas1p, notas2p, promedio, nombres_completos
        )

        recursados(promedio, nombres_completos)
        print("\n")
        apellidos_impresion(apellido_largo, su_nota)
        print("\n")
        promocion_impresion(alumnos_posibles)
        print("\n")

    except Exception:
        # if len(nombres) != len(apellidos) != len(notas1p) != len(notas2p): #VER QUE PONER ACA PARA QUE FUNCIONE
        print("No hay notas cargadas")
