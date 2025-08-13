
 
from random import randint
def strip(s):
  i = 0
  while i < len(s) and (s[i] == " " or s[i] == "\xa0" or s[i] == "\n"):
    i += 1
  s = s[i:][::-1]
  
  j = 0
  while j < len(s) and (s[j] == " " or s[j] == "\xa0" or s[j] == "\n"):
    j += 1
  s = s[j:][::-1]
  
  return s
  
 
 
 
 
def validacion_cant(num):
  try:
    OK = True
    if not (int(num) == num):
      OK = False
    if num <= 0:
      OK = False
    if OK:
      return True
  except:
    mensaje = "La cantidad debe ser positiva"
    return mensaje
 
 
      
 
 
 
 
def lectura_TXT(file):
  codigos = []
  cantidades = []
  try:
    with open(file, "r", encoding="utf-8") as archivo:
      for line in archivo:
        for i in range(len(line)):
          if line[i] == ":":
            pos = i
            codigo_producto = strip(line[:pos])
            cantidad_disponible = strip(line[pos + 1:])
            codigos.append(codigo_producto)
            cantidades.append(cantidad_disponible)
    return codigos,cantidades
  except FileNotFoundError:
    print("Error: no se ha encontrado el archivo")
    return None
 
 
 
 
 
 
 
 
def creacion_TXT(file):
  try:
    with open(file, "w", newline="", encoding="utf-8") as archivo:
      for codigo in range(1000,1010):
        archivo.write(f"{codigo}: 100\n")
  except FileNotFoundError:
    print("Error: no se ha encontrado el archivo")
    return None
 
 
 
 
 
 
 
 
def azar(file):
  L3 = []
  L4 = []
  L5 = []
  L6 = []
  L1,L2 = lectura_TXT(file)
  ln = len(L1)
  for i in range(len(L1)):
    eleccion = i
    eleccion = randint(0,len(L1) - 1)
    vendidos = randint(1,5)
    print(f"Producto {L1[eleccion]}: {vendidos} unidades ")
    L3.append(L1[eleccion])
    L4.append(vendidos)
  
  for elem in L3:
    elem = "-" + elem
    if elem not in L1:
      L1.append(elem)
 
  for j in range(len(L2)):
      actualizacion = int(L2[j]) - int(L4[j])
      L2[j] = actualizacion
  L5 = L1
  L6 = L2
  
  return L3,L4,L5,L6
 
 
 
 
 
 
def actualizacion_TXT(file):
  L1,L2 = lectura_TXT(file)
  L1,L2,L5,L6 = azar(file)
  try:
    with open(file, "a", newline="", encoding="utf-8") as archivo:
      for i in range(len(L5)):
        if i < len(L6)
          archivo.write(f"{L5[i]}: {L6[i]} \n")
  except FileNotFoundError:
    print("Error: no se ha encontrado el archivo")
    return None
 
 
 
 
  
 
 
 
 
def vender_productos(ruta, cantidad):
  validacion_cant(cantidad)
  lectura_TXT(ruta)
  if lectura_TXT(ruta) == None:
    creacion_TXT(ruta)
  azar(ruta)
  actualizacion_TXT(ruta)
 
 