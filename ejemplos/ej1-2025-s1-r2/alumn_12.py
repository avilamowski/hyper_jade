def validar(texto):
    if len(texto) == 0:
        return False


def split(texto, c):
    res = []
    last = 0
    for i in range(len(texto)):
        if texto[i] == c:
            if i - last > 0:
                res.append(texto[last:i])
            last = i + 1
    if len(texto) > last:
        res.append(texto[last:])
    return res


def strip(cadena):
    inicio, fin = 0, len(cadena) - 1
    while inicio <= fin and texto[inicio] == " ":
        inicio += 1
    while fin >= inicio and texto[fin] == " ":
        fin -= 1
    return texto[inicio : fin + 1]


def mostrar_lineas(texto):
    partes = []
    if not (validar(texto)):
        return "Texto Invalido"
    partes = split(texto, "\n")
    for i in range(len(partes)):
        if partes[i] != (" " * len(partes[i])):
            print(i + 1, ":", partes[i], sep="")


texto = "Hola  gente del universo\n \n123 \n Chau todos\n 90 adios"
mostrar_lineas(texto)
