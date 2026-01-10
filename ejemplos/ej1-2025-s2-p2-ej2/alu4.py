import csv
 
 
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
 
 
 
 
def venta_txt(archivo):
  cosa1 = []
  vendido = []
  try:
    with open("venta.txt", "r", encoding = "utf-8") as archivo:
      linea = archivo.read()
      for elem in linea:
        parte = split_custom(linea, ":")
        cosa1.append(parte[0])
        vendido.append(parte[1])
        
  except FileNotFoundError:
    return "0.0"
  
  return cosa1, vendido
 
 
def inventario_csv(archivo):
  cosa2 = []
  precio = []
  cantidad = []
   try:
     with open("inventario.csv", "r", encoding = "utf-8") as archivo:
       linea = archivo.read()
       for elem in linea:
         cosa2. append(linea[0])
         precio.append(linea[1])
         cantidad.append(linea[2])
    except FileNotFoundError:
      return "No existe el archivo"
  return cosa2, precio, cantidad
 
 
#1
def monto(cosa1, cosa2, vendido, cantidad, precio):
  monto = 0
  for i in range(len(cosa1)):
    if cosa1[i] == cosa2[i]:
      monto += precio*cantidad
  
  return monto
 
 
def costo(inventario_csv, venta_txt):
  cosa1, vendido = venta_txt()
  cosa2, cantidad, precio = inventario_csv()
  print(venta(txt()))    #por si el archivo no existe
  print(monto(cosa1, cosa2, vendido, cantidad, precio))
 
 
def cantidad_final(num):
  return num
 
 
def hacer_pedido(inventario_csv, venta_txt,cantidad_final):
  archivo = "inventario.csv"
  cant_inventario = inventario_csv(archivo)
  
  for i in cant_inventario:
    if float(cant_inventario[i]) < cantidad_final
    return []
  
 
 
print(costo(inventario_csv, venta_txt))
print(hacer_pedido(inventario_csv, venta_txt, cantidad_final))
 
 