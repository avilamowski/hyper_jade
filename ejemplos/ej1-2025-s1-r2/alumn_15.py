def validar(texto):
    try:
        valido = True
        j = 0
        while valido and j <= len(texto):
            for i in range(len(texto)):
                if "a" <= texto[i] <= "z":
                    valido = True
                elif "A" <= texto[i] <= "Z":
                    valido = True
                elif texto[i] == "\n":
                    valido = True
                elif texto[i] == " ":
                    valido = True
                elif "0" <= texto[i] <= "9":
                    valido = True
                else:
                    valido = False
            j += 1
    except:
        print("Ha ocurrido un error con el texto")
        valido = False
    return valido


def split_custom(texto, separador=" "):
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


def strip_custom(texto):
    inicio, fin = 0, len(texto) - 1
    while inicio <= fin and texto[inicio] == " ":
        inicio += 1
    while fin >= inicio and texto[fin] == " ":
        fin -= 1
    return texto[inicio : fin + 1]


def mostrar_lineas(texto):
    valido = validar(texto)
    if valido:
        lista_lineas = split_custom(texto, "\n")
        contador = 1
        for i in range(len(lista_lineas)):
            nro_fila = contador
            texto_fila = ""
            fila = strip_custom(lista_lineas[i])
            texto_fila += fila
            if texto_fila != "":
                print(nro_fila, ":", texto_fila)
            contador += 1
    else:
        print("Texto inválido")


def valido_numero(texto):
    try:
        numero = 0
        for i in range(len(texto)):
            numero += int(texto[i])
    except:
        numero = 0
    return numero


def promedio_numeros(texto):  # falta terminar!!
    todos_numeros = ""
    for i in range(len(texto)):
        j = 1
        valor = 0
        while j <= len(texto):
            partecita = texto[i:j]
            num = valido_numero(partecita)
            if num > 0:
                j += 1
                valor += num
        todos_numeros += str(valor)


def ordenar_len(lista):
    for i in range(len(lista)):
        for j in range(i + 1, len(lista)):
            if len(lista[i]) < len(lista[j]):
                lista[i], lista[j] = lista[j], lista[i]
    return lista


def sin_repetir(texto):
    valido = validar(texto)
    if valido:
        with open("sin_repeticiones.txt", "w", encoding="utf-8") as archivo:
            sin_repes = []
            lista_palabras = split_custom(texto, " ")
            for i in range(len(lista_palabras)):
                for j in range(i, len(lista_palabras)):
                    if lista_palabras[i] == lista_palabras[j]:
                        sin_repes.append(lista_palabras[i])
            escribir = str(ordenar_len(sin_repes))
            archivo.write(escribir)
    else:
        print("Texto inválido")


texto = "Hola1  gente del universo\n \n123 \n Chau todos\n 90 adios"
validar(texto)
mostrar_lineas(texto)
sin_repetir(texto)
