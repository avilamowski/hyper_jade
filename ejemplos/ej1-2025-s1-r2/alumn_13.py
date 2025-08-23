def validar(texto):
  vacio = True
  i = 0
  while i<len(texto):
    if (texto[i] != " ") and (texto[i] != ""):
      vacio = False
      if not ((97 <= ord(frase[i])<=122) or (65 <= ord(frase[i])<=90) or frase[i] in "1234567890 " or frase[i] == "\n"):
        vacio = True
      else:
        vacio = False
    i += 1
  if vacio == False:
    return True
  else:
    return False
def ordenar_custom(lista):
    for i in range(len(lista)):
        for j in range(i + 1, len(lista)):
            if len(lista[i]) < len(lista[j]):
                lista[i], lista[j] = lista[j], lista[i]
    return lista
def mostrar_lineas(frase):
  if validar(frase) == True:
    limpia = split_custom(strip_custom(frase), "\n")
    for i in range(len(limpia)):
      for j in range(len(limpia[i])):
        limpia = split_custom(strip_custom(frase), "\n")
    for i in range(len(limpia)):
      if limpia[i] != " ":
        limpia = strip_custom(limpia)
        print(i+1, ":", limpia[i])
  else:
    print("Texto inválido") 
 
def escribir(limpia):
  try:
    with open("sin_repeticiones.txt", "w") as archivo1:
      if validar(frase) == True:
        for l in limpia2:
          archivo1.write(l+"\n")
  except Exception:
    print("error")
frase = frase = "Hola  gente del universo\n \n123 \n Chau todos\n 90 adios"
valida = validar(frase)
mostrar = mostrar_lineas(frase)﻿
ordenada = ordenar_custom(frase)
escribir(ordenada)