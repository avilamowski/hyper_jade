# 1
def validar(texto):
    if texto != "":
        for letra in texto:
            if not (
                "a" <= letra <= "z"
                or "A" <= letra <= "Z"
                or "0" <= letra <= "9"
                or letra == "\n"
                or letra == " "
            ):
                return False
    else:
        return False
    return True


# 2
def mostrar_lineas(texto):
    if validar(texto):
        cont = 0
        nuevo_texto = ""
        for letra in texto:
            if letra == "\n":
                cont += 1
                if nuevo_texto and nuevo_texto != " ":
                    print(f"{cont}: {nuevo_texto}")
                    nuevo_texto = ""
            else:
                nuevo_texto += letra
        if nuevo_texto and nuevo_texto != " ":
            print(f"{cont+1}: {nuevo_texto}")
            nuevo_texto = ""
    else:
        print("Texto invalido")


# 3
def promedio_numeros(texto):
    if validar(texto):
        numero = 0
        cantidad = 0
        for letra in texto:
            if "0" <= letra <= "9":
                numero += int(letra)
                cantidad += 1

        if cantidad > 0:
            promedio = (numero / cantidad) * 100
            print(f"El promedio de los números en el texto es: {promedio:.2f}\n")
    else:
        print("Texto invalido")


texto = "Hola gente del universo\n \n123 \n Chau todos\n 90 adios"
promedio_numeros(texto)
mostrar_lineas(texto)

# la verdad que la consigna de como calcular el promedio no esta biene explicado, no se entiende
# tambien pense en esta forma len(texto)/5*100 pero tampoco da el resultado que dice ahi el #ejemplo
