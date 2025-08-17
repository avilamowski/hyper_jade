import random


def sacar_salto(lista):
    lista_nueva = []
    for linea in lista:
        if "\n" in linea:
            lista_nueva.append(linea[0 : len(linea) - 1])
        else:
            lista_nueva.append(linea)
    return lista_nueva


def split_custom(texto, separador):
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


def validar_cant(cantidad):
    try:
        if cantidad <= 0:
            raise ValueError("La cantidad debe de ser entero positivo")
    except ValueError as e:
        print(f"Error: {e}")
    except TypeError:
        print(f"Error: La cantidad debe de ser entero positivo")
    else:
        return cantidad


def abrir_archivo(archivo):
    try:
        with open(archivo, "r", encoding="UTF-8") as fd:
            lista_info = []
            cod_prod = []
            cant_disp = []
            for linea in fd:
                lista_info.append(linea)
                lista_info = sacar_salto(lista_info)
            for p in lista_info:
                partes = split_custom(p, ":")
                cod_prod.append(partes[0])
                cant_disp.append(partes[1])
    except FileNotFoundError:
        print("El archivo no existe, lo creamos")
        try:
            with open(archivo, "w", encoding="UTF-8") as fd:
                for i in range(10):
                    fd.write(f"100{i}: 100\n")
        except Exception:
            print("Hubo un error al crear el archivo")

        with open(archivo, "r", encoding="UTF-8") as fd:
            try:
                lista_info = []
                cod_prod = []
                cant_disp = []
                for linea in fd:
                    lista_info.append(linea)
                    lista_info = sacar_salto(lista_info)
                for p in lista_info:
                    partes = split_custom(p, ":")
                    cod_prod.append(partes[0])
                    cant_disp.append(partes[1])
            except Exception:
                print("Error con el archivo")
    return cod_prod, cant_disp


def escribir_archivo(
    archivo, cod_prod, cod_posi, cod_neg, cant_disp_posi, cant_disp_neg
):
    try:
        with open(archivo, "w", encoding="UTF-8") as fd:
            j = 0
            while j < len(cod_posi):
                fd.write(f"{cod_posi[j]}: {cant_disp_posi[j]}\n")
                j += 1
            a = 0
            while a < len(cod_neg):
                fd.write(f"{cod_neg[a]}: {cant_disp_neg[a]}\n")
                a += 1
    except Exception as e:
        print("Hubo un error al crear el archivo {e}")


def vender_productos(ruta, cantidad):
    cantidad = validar_cant(cantidad)
    cod_prod, cant_disp = abrir_archivo(ruta)
    cod_posi = []
    cod_neg = []
    cant_disp_posi = []
    cant_disp_neg = []
    cant_vend = 0
    for p in range(len(cod_prod)):
        if int(cod_prod[p]) > 0:
            cod_posi.append(cod_prod[p])
            cant_disp_posi.append(cant_disp[p])
        else:
            cod_neg.append(cod_prod[p])
            cant_disp_neg.append(cant_disp[p])
    i = 0
    while i < cantidad:
        cant_rand = random.randint(1, 5)
        pos_prod = random.randint(0, len(cod_posi) - 1)
        if int(cant_disp[pos_prod]) > cant_rand:
            cant_disp_posi[pos_prod] = int(cant_disp_posi[pos_prod]) - cant_rand
            cant_disp_neg = +cant_rand
        elif len(cant_disp_neg) == 0 and cod_disp_neg == 0:
            cant_disp_neg.append(cant_rand)
            cant_disp_neg.appned(-int(cod_posi))
        elif int(cant_disp_posi[pos_prod]) < cant_rand:
            cant_disp_posi[pos_prod] = 0
            cant_disp_neg[pos_prod] = int(cant_disp_neg[pos_prod]) + cant_rand
        i += 1
    escribir_archivo(ruta, cod_prod, cod_posi, cod_neg, cant_disp_posi, cant_disp_neg)


vender_productos("inventario.txt", 2)
