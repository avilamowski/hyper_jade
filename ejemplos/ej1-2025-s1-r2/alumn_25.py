def validar(texto):
  for caracter in range(len(texto)): 
    if not(texto[caracter]=="\n" or "0"<=texto[caracter]<="9" or (texto[caracter]>="A" and texto[caracter]<="Z") or (texto[caracter]>="a" and texto[caracter]<="z") or (texto[caracter] == " ")):
      return False
  return True
 
 
def split(texto, separador):
  subtextos = []
  subtexto = ""
  for caracter in texto:
    if caracter == separador:
      subtextos.append(subtexto)
      subtexto = ""
    else:
      subtexto += caracter
  subtextos.append(subtexto)
  return subtextos
 
 
def impresion_mostrarlineas(texto):
  for i in range(len(texto)):
    if not(texto[i] == "" or texto[i] == " "):#elimino cadenas vacias
      print(f"{i+1}: {texto[i]}") #Le sumo uno para que el orden sea no el de una lista sino el real, nosotros arrancamos de contar desde el 1, la maquina desde el 0
 
 
def mostrarlineas(texto):
  if validar(texto):
    texto_separado = split(texto, "\n")
    impresion_mostrarlineas(texto_separado)
  return texto_separado #me servira para "sin_repetir"
 
 
def promedio_numeros(lista):
  cadena_numerica = ""
  cantidad_numeros = 0
  sumatoria = 0
  for numero in lista:
    if "0"<=numero<="9":
      cadena_numerica += numero
    else:
      if cadena_numerica:
        cantidad_numeros+=1
        sumatoria+=int(cadena_numerica)
      cadena_numerica = ""
      
  try:
    print(sumatoria/cantidad_numeros)
  except ZeroDivisionError as e:
    print("No hay numeros en el texto")
  return
  
def escribirtxt(file, lista):
  try:
    with open(file, "w", encoding="utf-8") as archivo:
      for i in range(len(lista)):
        archivo.write(f"{lista[i]}")
    except FileNotFoundError as e:
      print("Problema creando el archivo")
    return
 
def sin_repetir(lista): #HAY QUE ARREGLAR, SEPARA BIEN, PERO NO LLEGA A HACER LA LISTA
  for i in range(0,len(lista)):
    lista_separada = split(lista[i], " ")
  print(lista_separada)  
 
 
texto =  "Hola  gente del universo\n \n123 \n Chau todos\n 90 adios"
texto_por_linea = mostrarlineas(texto)
promedio_numeros(texto)
sin_repetir(texto_por_linea)