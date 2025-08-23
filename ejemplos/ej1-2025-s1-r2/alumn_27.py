# ejericio 2
def validar(texto):
    for i in texto:
        if not (
            ("A" <= i <= "Z")
            or ("a" <= i <= "z")
            or ("0" <= i <= "9")
            or (i == " ")
            or (i == "\n")
        ):
            return False

    return True


texto = input("Ingrese un texto:")
val = validar(texto)


while texto != "":
    val = validar(texto)
    print(val)
    texto = input("Ingrese un texto:")

print("False")
texto = input("Ingrese un texto:")


def promedio_numeros(texto):
    cont = 0  # cuanats veces esta ese numero
    numeros = ""
    suma_num = 0

    for i in texto:
        if "0" <= i <= "9":
            numeros += i
            entero = int(i)
            suma_num += entero
            cont += 1

            print("La cantidad de numeros son:", cont)
            print("La suma total de num es:", suma_num)

    if cont > 0:
        promedio = suma_num / cont
        print("El promedio es:", promedio)

    return cont, numeros, suma_num


texto = input("Ingrese un texto:")
promedio_numeros(texto)
