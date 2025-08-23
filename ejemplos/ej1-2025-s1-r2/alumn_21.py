def validar(texto):
    if texto == "":
        return False
    if texto != str(texto):
        return False
    for i in texto:
        if not (
            "A" <= i <= "Z"
            or "a" <= i <= "z"
            or i == "\n"
            or i == " "
            or "0" <= i <= "9"
        ):
            return False
    return True


def split_enter(texto, separador="\n"):
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


def split_espacio(texto, separador=" "):
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


def un_espacio(texto):
    lista = split_espacio(texto)
    muestra = ""
    c = 0
    for i in lista:
        muestra += i
        if c < len(lista) - 1:
            muestra += " "
        c += 1
    return muestra


def mostrar_lineas(texto):
    if validar(texto):
        lista = split_enter(texto)
        c = 1
        for frase in lista:
            if frase != " ":
                muestra = strip_custom(frase)
                final = un_espacio(muestra)
                print(f"{c}: {final}")
            c += 1
    else:
        print("Texto invalido")


def sacanumeros(texto):
    nums = []
    text = split_espacio(texto)
    for i in text:
        num = ""
        for j in i:
            if "0" <= j <= "9":
                num += j
        if num != "":
            nums.append(int(num))
    return nums


def promedio_numeros(texto):
    if validar(texto):
        nums = sacanumeros(texto)
        total = 0
        for i in nums:
            total += i
        if len(nums) > 0:
            prom = total / len(nums)
            print(f"El promedio de los números en el texto es {prom}")
        else:
            print("Su texto no contiene numeros")
    else:
        print("Texto invalido")


def esolonumero(texto):
    letras = ""
    for i in texto:
        if "A" <= i <= "Z" or "a" <= i <= "z":
            letras += i
    if letras != "":
        return True
    else:
        return False


def texto_en_min(texto):
    text = ""
    for c in texto:
        if "A" <= c <= "Z":
            text += chr(ord(c) + 32)
        else:
            text += c
    return text


def sacapalabras(texto):
    palabras = []
    text = split_espacio(texto)
    for i in text:
        palabra = ""
        for j in i:
            if "A" <= j <= "Z" or "a" <= j <= "z" or "0" <= j <= "9":
                palabra += j
        palabra = texto_en_min(palabra)
        if palabra != "" and esolonumero(palabra) and palabra not in palabras:
            palabras.append(palabra)
    return palabras


def ordenar_custom(lista):
    for i in range(len(lista)):
        for j in range(i + 1, len(lista)):
            if lista[i][1] < lista[j][1]:
                lista[i], lista[j] = lista[j], lista[i]
    return lista


# Este punto no especifica, pero voy a tomar como palabras únicamente
# aquellos caracteres que sean letras del abecedario ingles
# No menciona que tienen que estar igual que de la forma escrita
# entonces, como lo paso a minuscula para comparar que no esté la palabra,
# en el txt  voy a escribir las palabras en minuscula
def sin_repetir(texto):
    if validar(texto):
        lista = sacapalabras(texto)
        len = []
        for i in lista:
            largo = 0
            for j in i:
                largo += 1
            len.append(largo)
        c = 0
        largos = []
        for j in lista:
            largos.append([j, len[c]])
            c += 1

        ordenados = ordenar_custom(largos)

        with open("sin_repeticiones.txt", "w", encoding="utf-8") as f:
            for i in ordenados:
                f.write(i[0] + "\n")
    else:
        return None
