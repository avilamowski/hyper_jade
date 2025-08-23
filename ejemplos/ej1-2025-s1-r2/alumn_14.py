def convertir_texto_en_lista(texto):
  lista = []
  for i in range(len(texto)):
    lista.append(texto[i])
  return lista
#la siguiente funcion retorna siempre False, esto validado mal, entonces se va a arrastrar el #error en las otras funciones, no me alcanzo el tiempo para corregirlo.
def recorrer_lista(lista):
  letras = convertir_texto_en_lista("abcdefghijklmnopqrstuvwxyz")
  numeros = convertir_texto_en_lista("1234567890")
  for elemento in lista:
    if elemento not in letras or numeros or [' ', '/n']:
      return False
  return True
﻿
def validar(texto):
  lista = convertir_texto_en_lista(texto)
  if lista == []:
    return False
  return recorrer_lista(lista)
 
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
 
def eliminar_espacios(lista):
  for i in range(len(lista)-1):
    if lista[i] == " " and lista[i+1] == " ":
      del lista[i+1]
    if lista[len(lista)] == " ":
      del(lista[len(lista)])
  return lista
  
def mostrar_lineas(texto):
  if validar(texto)==True:
    lista = split_custom(texto, separador="/n")
    eliminar_espacios(lista)
    cont_lineas = 0
    while cont_lineas < len(lista):
      for elemento in lista:
        if elemento == [" "]:
          cont_lineas +=1
        else:
          print(cont_lineas, end="")
          print(":", end="")
          print(elemento)
  else:
    print("Texto invalido")
 
def promedio_numeros(texto):
  if validar(texto) == True:
    lista = split_custom(texto, separador=" ")
    total_num = 0
    cant_num = 0
    for elemento in lista:
      if elemento in convertir_texto_en_lista(1234567890):
        totatl_num += elemento
        total_num +=1
    promedio = total_num / total_num
  else:
    print("Texto invalido")

