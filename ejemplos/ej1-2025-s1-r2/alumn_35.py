def split_custom1(texto, separador="\n"):
    resultado = []
    palabra = ""
    for char in texto:
        if char == separador:
            if palabra:
                resultado.append(palabra)
                palabra = ""
        else:
            palabra += char
    if palabra:
        resultado.append(palabra)
    return resultado


def contar_elem(lista):
    lista_orden = []
    for elemento in range(1, len(lista) + 1):
        lista_orden.append(elemento)
    return lista, lista_orden


def split_custom2(texto, separador):
    resultado = []
    palabra = ""
    for char in texto:
        if char == separador:
            if palabra:
                resultado.append(palabra)
                palabra = ""
        else:
            palabra += char
    if palabra:
        resultado.append(palabra)
    return resultado


def eliminar_vacio(lista1, lista2):
    lista1_nueva = []
    lista2_nueva = []
    for elem in range(len(lista1)):
        if lista1[elem] != " ":
            lista1_nueva.append(lista1[elem])
            lista2_nueva.append(lista2[elem])
    return lista1_nueva, lista2_nueva


def mostrar_lineas(texto):
    lista_de_texto = split_custom1(texto)
    palabras = contar_elem(texto)
    lista_texto_limpia, palabras_limpia = eliminar_vacio(lista_de_texto, palabras)
    for elem in range(len(lista_texto_limpia, palabras_limpia)):
        linea = palabras_limpia[elem]
        frase = lista_texto_limpia[elem]
        print(f"{linea}: {frase}")
