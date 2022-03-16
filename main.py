import csv
from datetime import datetime
import certifi
import ssl
import geopy.geocoders
import numpy as np
import cv2 as cv
import os
from geopy.units import kilometers
from geopy.distance import geodesic

ctx = ssl.create_default_context(cafile=certifi.where())
geopy.geocoders.options.default_ssl_context = ctx

geolocator = geopy.geocoders.Nominatim(user_agent='hola')


def amarillo(imagen, imagen_hsv):
    """
    >Pre: toma la imagen original y la imagen HSV. Evalua el porcentaje de verde en la imagen.
    >Post: Si el porcentanje es mayor a 0.07 entonces hay verde, devuelve True
    """
    hay_amarillo: bool

    # Rango necesario para detectar el amarillo
    amarillo_bajo = np.array([20, 100, 20], np.uint8)
    amarillo_alto = np.array([32, 255, 255], np.uint8)
    mask_hsv = cv.inRange(imagen_hsv, amarillo_bajo, amarillo_alto)

    mask = cv.inRange(imagen, amarillo_bajo, amarillo_alto)
    resultado_imagen = cv.bitwise_and(imagen, imagen, mask=mask_hsv)
    cant_amarillo = np.sum(resultado_imagen) / np.sum(imagen_hsv)

    if cant_amarillo > 0.07:
        hay_amarillo = True
    else:
        hay_amarillo = False

    return hay_amarillo


def azul(imagen, imagen_hsv):
    """
    >Pre: toma la imagen original y la imagen HSV. Evalua el porcentaje de azul en la imagen.
    >Post: Si el porcentanje es mayor a 0.07 entonces hay azul, devuelve True
    """
    hay_azul: bool

    # Rango necesario para detectar el azul
    azul_bajo = np.array([100, 100, 20], np.uint8)
    azul_alto = np.array([125, 255, 255], np.uint8)
    mask_hsv = cv.inRange(imagen_hsv, azul_bajo, azul_alto)

    mask = cv.inRange(imagen, azul_bajo, azul_alto)
    resultado_imagen = cv.bitwise_and(imagen, imagen, mask=mask_hsv)
    cant_azul = np.sum(resultado_imagen) / np.sum(imagen_hsv)

    if cant_azul > 0.07:
        hay_azul = True
    else:
        hay_azul = False

    return hay_azul


def rojo(imagen, imagen_hsv):
    """
    >Pre: toma la imagen original y la imagen HSV. Evalua el porcentaje de rojo en la imagen.
    >Post: Si el porcentanje es mayor a 0.07 entonces hay rojo, devuelve True
    """
    hay_rojo: bool
    # Rango necesario para detectar el rojo

    rojo_bajo_uno = np.array([0, 100, 20], np.uint8)
    rojo_alto_uno = np.array([10, 255, 255], np.uint8)

    rojo_bajo_dos = np.array([175, 100, 20], np.uint8)
    rojo_alto_dos = np.array([80, 255, 255], np.uint8)

    rojo_uno = cv.inRange(imagen_hsv, rojo_bajo_uno, rojo_alto_uno)
    rojo_dos = cv.inRange(imagen_hsv, rojo_bajo_dos, rojo_alto_dos)

    mask_hsv = cv.add(rojo_uno, rojo_dos)
    mask = cv.inRange(imagen, cv.add(rojo_bajo_uno, rojo_bajo_dos), cv.add(rojo_alto_uno, rojo_alto_dos))

    resultado_imagen = cv.bitwise_and(imagen, imagen, mask=mask_hsv)
    cant_rojo = np.sum(resultado_imagen) / np.sum(imagen_hsv)

    if cant_rojo > 0.07:
        hay_rojo = True

    else:
        hay_rojo = False

    return hay_rojo


def verde(imagen, imagen_hsv):
    """
       >Pre: toma la imagen original y la imagen HSV. Evalua el porcentaje de verde en la imagen.
       >Post: Si el porcentanje es mayor a 0.07 entonces hay verde, devuelve True
    """
    hay_verde: bool
    # Rango necesario para detectar el verde

    verde_bajo = np.array([36, 100, 20], np.uint8)
    verde_alto = np.array([80, 255, 255], np.uint8)
    mask_hsv = cv.inRange(imagen_hsv, verde_bajo, verde_alto)

    mask = cv.inRange(imagen, verde_bajo, verde_alto)
    resultado_imagen = cv.bitwise_and(imagen, imagen, mask=mask_hsv)
    cant_verde = np.sum(resultado_imagen) / np.sum(imagen_hsv)

    if cant_verde > 0.07:
        hay_verde = True
    else:
        hay_verde = False

    return hay_verde


def negro(imagen, imagen_hsv):
    """
    >Pre: toma la imagen original y la imagen HSV. Evalua el porcentaje de negro en la imagen.
    >Post: Si el porcentanje es mayor a 0.07 entonces hay negro, devuelve True
    """
    hay_negro: bool

    cant_negro = np.sum(imagen) / np.sum(imagen_hsv)

    if cant_negro > 0.07:
        hay_negro = True
    else:
        hay_negro = False

    return hay_negro


def detectar_color(imagen) -> str:
    """
    >Pre: recibe la imagen y pasa por todos los colores.
    >Post: una vez que encuentra el color devuelve el color que es
    """

    imagen_hsv = cv.cvtColor(imagen, cv.COLOR_BGR2HSV)
    color_encontrado: str = ""

    if amarillo(imagen, imagen_hsv) is True:
        color_encontrado = "Amarillo"
    elif azul(imagen, imagen_hsv) is True:
        color_encontrado = "Azul"
    elif rojo(imagen, imagen_hsv) is True:
        color_encontrado = "Rojo"
    elif verde(imagen, imagen_hsv) is True:
        color_encontrado = "Verde"
    elif negro(imagen, imagen_hsv) is True:
        color_encontrado = "Negro"

    return color_encontrado


def cargar_yolo():
    """
       > Cargo los archivos necesarios para hacer uso de la red y de yolo
       """

    # Cargo el archivo de configuracion y el archivo weights para armar la red
    nombre_archivo: str = "yolov3.cfg"
    archivo: str = "yolov3.weights"
    configuracion_archivo = os.path.dirname(os.path.abspath(__file__)) + '/' + nombre_archivo
    weights_archivo = os.path.dirname(os.path.abspath(__file__)) + '/' + archivo
    red = cv.dnn.readNetFromDarknet(configuracion_archivo, weights_archivo)

    clases = []

    # Cargo los nombres de las clases (Los objetos que la red fue entrenada para identificar)
    c_archivo: str = "coco.names"
    clases_archivo = os.path.dirname(os.path.abspath(__file__)) + '/' + c_archivo

    with open(clases_archivo, 'rt') as f:
        clases = [linea.strip() for linea in f.readlines()]

    output_capas = [layer_name for layer_name in red.getUnconnectedOutLayersNames()]

    colores = np.random.uniform(0, 255, size=(len(clases), 3))

    return red, clases, colores, output_capas


def cargar_imagen(img_path):
    """
    >Pre: recibe el path de la imagen
    >Post: devuelve la imagen reajustada
    """
    img = cv.imread(img_path)
    img = cv.resize(img, None, fx=0.4, fy=0.4)
    altura, ancho, channels = img.shape

    return img, altura, ancho, channels


def detectar_objetos(img, red, outputLayers):
    """
    >Pre: la funcion recibe 3 "variables" img  es un arreglo de n-dimensionds, Red es la red neuronal, y outputlayers
         una lista
    >Post: devuelve toda la informacion acerca de todos los objetos detectados (coordenadas, confianza en base a todas
         clases que hay en coco.names, etc).
    """
    # Construyo BLOB  de la imagen
    blob = cv.dnn.blobFromImage(img, 1 / 255.0, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)

    # Input a la red
    red.setInput(blob)
    salidas = red.forward(outputLayers)

    return blob, salidas


def get_box_dimensions(outputs, altura, ancho):
    """
    > De acuerdo a los parametros obtenidos, determina el nivel de confianza, clases a las que podria corresponder
     la imagen y el bounding box
    """
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detectar in output:
            scores = detectar[5:]

            # Se identifica el indice de clase con la confianza(confidence score) mas alto
            class_id = np.argmax(scores)
            conf = scores[class_id]

            # Seleccionamos los cuadros de prediccion con una confianza de mas del 30%
            if conf > 0.3:
                center_x = int(detectar[0] * ancho)
                center_y = int(detectar[1] * altura)
                w = int(detectar[2] * ancho)
                h = int(detectar[3] * altura)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)

    return boxes, confs, class_ids


def marcar_etiquetas(boxes, confs, class_ids, clases, colores, img) -> str:
    """
    > Funcion que se encarga de etiquetar la imagen en base a la informacion obtenida (como por ejemplo el porcentaje
      de confianza)
    """
    indexes = cv.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv.FONT_HERSHEY_PLAIN

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            color = colores[i]
            etiqueta = str(clases[class_ids[i]])

    return etiqueta


def detectar_imagen(img_path) -> str:
    """
    > En base a la informacion recopilada de las otras funciones
    > Nos devuelve la etiqueta de la imagen, es decir la imagen identificada
    """
    model, clases, colores, output_capas = cargar_yolo()
    imagen, altura, ancho, channels = cargar_imagen(img_path)
    blob, outputs = detectar_objetos(imagen, model, output_capas)
    boxes, confs, class_ids = get_box_dimensions(outputs, altura, ancho)
    etiqueta = marcar_etiquetas(boxes, confs, class_ids, clases, colores, imagen)

    return etiqueta


def asociar_color_articulo(dic_stock: dict, img_path, imagen) -> None:
    """
    >Pre: Recibe el diccionario vacio y en base a la etiqueta recibida en la funcion detectar_imagen
     modifica el diccionario clasificando los articulos encontrados.
    >Post: En caso de que se haya encontrado un animal avisa que el proceso se detiene
    """
    # codigo de articulo: -1334 botellas -568 vasos
    etiqueta = detectar_imagen(img_path)

    if etiqueta != "bottle" and etiqueta != "cup":
        # En caso de que detecte un animal el proceso se "detiene"
        print("PROCESO DETENIDO! Se reanuda en 1 minuto! ")

    else:
        color_encontrado: str = detectar_color(imagen)
        codigo_art: int = 0
        if etiqueta == "bottle":
            codigo_art = 1334
        elif etiqueta == "cup":
            codigo_art = 568

        if codigo_art not in dic_stock.keys():
            dic_stock[codigo_art] = {}
            dic_stock[codigo_art][color_encontrado] = 1

        elif codigo_art in dic_stock.keys():
            if color_encontrado not in dic_stock[codigo_art].keys():
                dic_stock[codigo_art][color_encontrado] = 1
            else:
                dic_stock[codigo_art][color_encontrado] += 1


def clasificar_stock() -> dict:
    """
        > Pre: recorre la carpeta de imagenes
        > Post: devuelve un diccionario con el stock disponible clasficado por articulo (codigo de articulo)
          y cantidad disponible de c/color
    """
    dic_stock: dict = {}
    carpeta: str = "Lote0001"

    carpeta_imagenes_path = os.path.dirname(os.path.abspath(__file__)) + '/' + carpeta

    try:
        nombre_archivos = os.listdir(carpeta_imagenes_path)
    except:
        print('No se encontró la carpeta con imágenes')
    else:
        for nombre_archivo in nombre_archivos:
            imagen_path = carpeta_imagenes_path + "/" + nombre_archivo
            imagen = cv.imread(imagen_path)
            asociar_color_articulo(dic_stock, imagen_path, imagen)

    return dic_stock


def colores_disponbles(stock: dict) -> dict:
    """
    > A partir del stock disponible nos devuelve un diccionario mas simple con la disponibilidad de colores
      de cada articulo
    """
    colores_dispo: dict = {}
    for codigo, valor in stock.items():
        for color in valor.keys():
            if codigo not in colores_dispo.keys():
                colores_dispo[codigo] = [color]
            else:
                if color not in colores_dispo[codigo]:
                    colores_dispo[codigo].append(color)

    return colores_dispo


def crear_archivos(stock: dict):
    """
    > Pre: Toma la disponibilidad de stock
    > Post: Crea un archivo txt para c/articulo con los colores y la cantidad disponible de c/color
    """
    codigo_botella: int = 1334
    codigo_vaso: int = 568

    for clave, valor in stock.items():
        if clave == codigo_botella:
            with open("botellas.txt", "w") as agendaarchivo:
                for nombre, valor in stock[clave].items():
                    agendaarchivo.write("%s %s\n" % (nombre, valor))
        elif clave == codigo_vaso:
            with open("vasos.txt", "w") as agendaarchivo:
                for nombre, valor in stock[clave].items():
                    agendaarchivo.write("%s %s\n" % (nombre, valor))


def lectura_pedidos(archivo: str) -> dict:
    """
    Precondicion:Un archivo pedidos.csv que contiene por linea un articulo por pedido

    Postcondicion: lee el archivo y  retorna el diccionario pedido actualizado
                Pedidos es nuestro diccionario =
                {Nro. Pedido(tipo entero): {Fecha: dato,  Cliente: cadena,
                                         Ciudad: cadena,  Provincia:cadena,
                                         Articulos:{Cod.Artículo(entero):
                                                    {Color: cadena, Cantidad: entero, descuento:entero }}}}
    """
    datos = list()
    pedidos: dict = {}
    try:
        pedidos_csv = open(archivo, newline='', encoding="UTF-8")
    except:
        print('No hay pedidos cargados')
        with open(archivo, 'a') as pedidos_csv:
            pedidos_csv.write(
                'Nro. Pedidio, Fecha, Cliente, Ciudad, Provincia, Cod. Artículo, Color, Cantidad, Descuento\n')
    else:
        lector = csv.reader(pedidos_csv, delimiter=',')
        next(lector)
        for row in lector:
            datos.append(row)

        for dato in datos:
            if int(dato[0]) not in pedidos:
                pedidos[int(dato[0])] = {'Fecha': dato[1], 'Cliente': dato[2], 'Ciudad': dato[3], 'Provincia': dato[4],
                                         'Artículos': {int(dato[5]): {
                                             dato[6]: {'Cantidad': int(dato[7]), 'Descuento': int(dato[8])}}}}
            elif int(dato[5]) in pedidos[int(dato[0])]['Artículos']:
                pedidos[int(dato[0])]['Artículos'][int(dato[5])][dato[6]] = {'Cantidad': int(dato[7]),
                                                                             'Descuento': int(dato[8])}
            else:
                pedidos[int(dato[0])]['Artículos'][int(dato[5])] = {
                    dato[6]: {'Cantidad': int(dato[7]), 'Descuento': int(dato[8])}}
        pedidos_csv.close()

    return pedidos


def carga_articulos() -> dict:
    """
    Precondicion: Se le pide al usuario ingresar los datos del articulo.
    Postcondicion: Con los datos ingresados por el usuario, se creay retorna un diccionario
                Articulos:{Cod. Artículo(entero): {Color: cadena, Cantidad: entero, descuento:entero }}}}
                las claves de Articulos seran codigo de articulo(tipo entero),y las claves del diccionario por tipo de
                articulo es el color (cadena)
    """

    art_disp: dict = {1334: ['Azul', 'Amarillo', 'Rojo', 'Negro', 'Verde'], 568: ['Azul', 'Negro']}
    respuesta: str = 'si'
    articulos: dict = {}
    cant: int = 1
    while respuesta == 'si':
        codigo: str = input('Ingrese el código del artículo: ')
        valido: bool = False
        while not valido:
            while not codigo.isnumeric():
                print('INGRESO INVÁLIDO')
                codigo: str = input('Ingrese el código del artículo: ')
            if int(codigo) in art_disp:
                valido = True
            else:
                codigo = ''

        articulos[int(codigo)] = {}
        respuesta_color: str = 'si'
        while respuesta_color == 'si':
            color: str = input('Ingrese el color: ')
            while color not in art_disp[int(codigo)]:
                print('Este color no es válido')
                color: str = input('Ingrese el color: ')
            cantidad: str = input('Ingrese la cantidad: ')
            while not cantidad.isnumeric():
                print('INGRESO INVÁLIDO')
                cantidad: str = input('Ingrese la cantidad: ')
            cantidad = cantidad
            descuento: str = input('Ingrese el descuento: ')
            while not descuento.isnumeric():
                print('INGRESO INVÁLIDO')
                descuento: str = input('Ingrese la descuento: ')
            descuento = descuento
            articulos[int(codigo)][color] = {'Cantidad': int(cantidad), 'Descuento': int(descuento)}
            print(
                'Quiere agregar otro color? Si agrega un color que ya ingresó el nuevo ingreso sobreescribirá el viejo')
            respuesta_color = input("Responda 'si' ó 'no': ")
            while respuesta_color != 'si' and respuesta_color != 'no':
                print('INGRESO INVÁLIDO')
                respuesta_color = input("Quiere agregar otro color? Responda 'si' ó 'no': ")
        print('Quiere agregar otro codigo? Si agrega un código que ya ingresó el nuevo ingreso sobreescribirá el viejo')
        respuesta = input("Responda 'si' ó 'no': ")
        while respuesta != 'si' and respuesta != 'no':
            print('INGRESO INVÁLIDO')
            respuesta = input("Quiere agregar otro codigo? Responda 'si' ó 'no': ")

    return articulos


def ingresar_pedido(pedidos: dict) -> dict:
    """
    Precondicion:Pedidos es nuestro diccionario =
                {Nro. Pedido(tipo entero): {Fecha: dato,  Cliente: cadena,
                                            Ciudad: cadena,  Provincia:cadena,
                                            Articulos:{Cod.Artículo(entero):
                                                            {Color: cadena, Cantidad: entero, descuento:entero }}}}
    Postcondicion: se retorna el diccionario pedido con los valores ingresados por el usuario.
    """
    numeros = list(map(int, pedidos.keys()))
    if len(numeros) > 0:
        numeros.sort()
        numero_pedido: int = int(numeros[-1]) + 1
    else:
        numero_pedido = 1

    fecha: str = datetime.today().strftime('%d/%m/%Y')

    cliente: str = input('Ingrese el nombre del cliente: ')
    ciudad: str = input('Ingrese la ciudad: ')
    provincia: str = input('Ingrese la provincia: ')
    articulos: dict = carga_articulos()
    with open("pedidos.csv", "a") as archivo:

        for articulo in articulos:
            for color in articulos[articulo]:
                cantidad: str = str(articulos[articulo][color]['Cantidad'])
                descuento: str = str(articulos[articulo][color]['Descuento'])
                archivo.write(
                    str(numero_pedido) + ',' + fecha + ',' + cliente + ',' + ciudad + ',' + provincia + ',' + str(
                        articulo) + ',' + color + ',' + cantidad + ','
                    + descuento + '\n')

    pedidos[numero_pedido] = {'Fecha': fecha, 'Cliente': cliente, 'Ciudad': ciudad, 'Provincia': provincia,
                              'Artículos': articulos}

    return pedidos


def modificar_pedido(pedidos: dict) -> dict:
    """
    Precondicion:Pedidos es nuestro diccionario =
                {Nro. Pedido(tipo entero): {Fecha: dato,  Cliente: cadena,
                                            Ciudad: cadena,  Provincia:cadena,
                                            Articulos:{Cod.Artículo(entero):
                                                            {Color: cadena, Cantidad: entero, descuento:entero }}}}
    Postcondicion: se retorna el diccionario pedido actualizado, con los cambios del usuario, tambien actualiza
                    el archivo pedidos.csv, todo con los valores ingresados por el usuario. elimina el pedido
                    previamente modificado en el archivo
    """
    pedido_a_modificar: str = input('Ingrese el numero del pedido que desee modificar: ')
    valido: bool = False
    while not valido:
        while not pedido_a_modificar.isnumeric():
            print('INGRESO INVÁLIDO')
            pedido_a_modificar: str = input('Ingrese el numero del pedido que desee modificar: ')
        if int(pedido_a_modificar) in pedidos:
            valido = True
        else:
            pedido_a_modificar = ''

    pedido_a_modificar = int(pedido_a_modificar)

    respuesta: str = 'si'
    while respuesta == 'si':
        print('Fecha: ', pedidos[pedido_a_modificar]['Fecha'])
        print('Cliente: ', pedidos[pedido_a_modificar]['Cliente'])
        print('Ciudad: ', pedidos[pedido_a_modificar]['Ciudad'])
        print('Provincia: ', pedidos[pedido_a_modificar]['Provincia'])
        print('Artículos: ')
        for articulo in pedidos[pedido_a_modificar]['Artículos']:
            for color in pedidos[pedido_a_modificar]['Artículos'][articulo]:
                print('- Color: ', color)
                print('        - Cantidad: ', pedidos[pedido_a_modificar]['Artículos'][articulo][color]['Cantidad'])
                print('        - Descuento: ', pedidos[pedido_a_modificar]['Artículos'][articulo][color]['Descuento'])

        dato_a_modificar: str = input('Ingrese el dato que desee modificar: ')

        while dato_a_modificar not in pedidos[pedido_a_modificar]:
            print('INGRESO INVÁLIDO')
            dato_a_modificar: str = input('Ingrese el dato que desee modificar: ')

        if dato_a_modificar != 'Artículos':

            modificacion: str = input('Ingrese su modificación: ')
            pedidos[pedido_a_modificar][dato_a_modificar] = modificacion

        else:
            modificacion: dict = carga_articulos()
            pedidos[pedido_a_modificar][dato_a_modificar] = modificacion
        respuesta = input("Quiere modificar otro dato de este pedido? Responda 'si' ó 'no': ")
        while respuesta != 'si' and respuesta != 'no':
            print('INGRESO INVÁLIDO')
            respuesta = input("Quiere modificar otro dato de este pedido? Responda 'si' ó 'no': ")

    with open("pedidos.csv", "r+") as f:
        lines = f.readlines()
        f.seek(0)
        for line in lines:
            clave = line[:line.index(",")]

            if str(pedido_a_modificar) != clave:
                f.write(line)
        f.truncate()
        articulos: dict = pedidos[pedido_a_modificar]['Artículos']
        for articulo in articulos:
            print('Código: ', articulo)
            for color in articulos[articulo]:
                f.write(str(pedido_a_modificar) + ',' + pedidos[pedido_a_modificar]['Fecha'] + ',' +
                        pedidos[pedido_a_modificar]['Cliente'] + ',' + pedidos[pedido_a_modificar]['Ciudad'] + ',' +
                        pedidos[pedido_a_modificar]['Provincia'] + ',' + str(articulo) + ',' + color + ',' + str(
                    articulos[articulo][color]['Cantidad']) + ',' + str(articulos[articulo][color]['Descuento']) + '\n')

    return pedidos


def eliminar_pedido(pedidos: dict) -> dict:
    """
    Precondicion:Pedidos es nuestro diccionario =
                {Nro. Pedido(tipo entero): {Fecha: dato,  Cliente: cadena,
                                         Ciudad: cadena,  Provincia:cadena,
                                         Articulos:{Cod.Artículo(entero):
                                                    {Color: cadena, Cantidad: entero, descuento:entero }}}}
    Postcondicion: se retorna el diccionario pedido actualizado, sin el pediddo eliminado, para eliminar un pedido
                    se le pide al usuario el numero de pedido, si existe, elimina  las lineas del archivo que contengan
                    el codigo, y ademas de eliminar el pedido del diccionario.
    """
    pedido_a_eliminar: str = input('Ingrese el numero del pedido que desee eliminar: ')
    valido: bool = False
    while not valido:
        while not pedido_a_eliminar.isnumeric():
            print('INGRESO INVÁLIDO')
            pedido_a_eliminar: str = input('Ingrese el numero del pedido que desee eliminar: ')
        if int(pedido_a_eliminar) in pedidos:
            valido = True
        else:
            pedido_a_eliminar = ''

    with open("pedidos.csv", "r+") as f:
        lines = f.readlines()
        f.seek(0)
        for line in lines:
            clave = line[:line.index(",")]

            if pedido_a_eliminar != clave:
                f.write(line)
        f.truncate()

    pedidos.pop(int(pedido_a_eliminar))

    return pedidos


def abm(pedidos: dict) -> dict:
    opciones: list = ["Agregar Pedido", "Modificar Pedido", "Borrar Pedido", "Salir"]
    opcion: str = ''

    while opcion != '4':

        print("Menu: ")
        for indice in range(len(opciones)):
            print(indice + 1, "- ", opciones[indice])

        opcion = input(" ")

        if opcion == '1':

            pedidos = ingresar_pedido(pedidos)

        elif opcion == '2':
            if len(pedidos) != 0:
                pedidos = modificar_pedido(pedidos)
            else:

                print('el diccionario esta vacio')

        elif opcion == '3':

            if len(pedidos) != 0:

                pedidos = eliminar_pedido(pedidos)
                print(pedidos)

            else:

                print('el diccionario esta vacio')

        elif opcion == '4':

            print("Saliste ABM")

        else:

            print("Las opciones deben ser entre 1 y 4")

    return pedidos


def lectura_stock(archivo: str) -> dict:
    datos = list()

    with open(archivo, newline='', encoding="UTF-8") as stock_csv:
        lector = csv.reader(stock_csv, delimiter=',')
        next(lector)
        for row in lector:
            datos.append(row)

    stock: dict = {}

    for dato in datos:
        if int(dato[0]) in stock:
            stock[int(dato[0])][dato[1]] = int(dato[2])
        else:
            stock[int(dato[0])] = {dato[1]: int(dato[2])}

    return stock


def disponible(stock: dict, codigo_articulo: str, color: str, cantidad: int) -> bool:
    """
    PRECONDICIONES: stock es un diccionario con el stock disponible de vasos y botellas con el formato
                       {1334: {'Azul': cantidad(int), 'Rojo': cantidad(int),..},
                            568:{'Azul': cantidad(int), 'Negro': cantidad(int)}}
                       codigo_atriculo es un entero, color es una cadena y cantidad es un entero positivo

    POSTCONDICIONES: se fija en el diccionario stock si el artículo está disponible y retorna True si si y False si no
    """

    if codigo_articulo in stock:
        if color in stock[codigo_articulo]:
            if cantidad <= int(stock[codigo_articulo][color]):
                return True
    else:
        return False


def pedidos_completos(pedidos: dict, stock: dict) -> list:
    """
        PRECONDICIONES: pedidos es un diccionario que contiene los pedidos realizados con el formato
                       {numero de pedido (int): {'Fecha': fecha(DD/MM/AAAA) (str), 'Cliente': nombre (str),
                       'Ciudad': ciudad (str), 'Provincia': provincia (str), 'Artículos': {1334:{color(str):
                       {'Cantidad': cantidad(int), 'Descuento': descuento(int)},... },568:{color(str): {'Cantidad':
                       cantidad(int), 'Descuento': descuento(int)},... }}},...},
                       stock es un diccionario con el stock disponible de vasos y botellas con el formato
                       {1334: {'Azul': cantidad(int), 'Rojo': cantidad(int),..}, 568:{'Azul': cantidad(int),
                       'Negro': cantidad(int)}}
       POSTCONDICIONES: Retorna una lista con los números (enteros) de los pedidos que se pueden completar con el
                        stock disponible.
    """

    stock_local: dict = {}

    for articulo in stock:
        for color in stock[articulo]:
            if articulo in stock_local:
                stock_local[articulo][color] = stock[articulo][color]
            else:
                stock_local[articulo] = {color: stock[articulo][color]}

    fechas: dict = {}

    completados: list = []

    for pedido in pedidos:
        fechas[pedido] = pedidos[pedido]['Fecha']

    ordenados = sorted(fechas.items(), key=lambda x: datetime.strptime(x[1], '%d/%m/%Y'))

    hay_stock: bool = True
    i: int = 0

    while len(list(stock_local.keys())) != 0 and i < len(ordenados):
        hay_stock: bool = True
        for articulo in pedidos[ordenados[i][0]]['Artículos']:
            for color in pedidos[ordenados[i][0]]['Artículos'][articulo]:
                if not disponible(stock_local, articulo, color,
                                  int(pedidos[ordenados[i][0]]['Artículos'][articulo][color]['Cantidad'])):
                    hay_stock = False

        if hay_stock:
            for articulo in pedidos[ordenados[i][0]]['Artículos']:
                for color in pedidos[ordenados[i][0]]['Artículos'][articulo]:
                    stock_local[articulo][color] = stock_local[articulo][color] - \
                                                   pedidos[ordenados[i][0]]['Artículos'][articulo][color]['Cantidad']
                    if stock_local[articulo][color] <= 0:
                        (stock_local[articulo]).pop(color)
                        if len(list((stock_local[articulo]).keys())) == 0:
                            stock_local.pop(articulo)
            completados.append(ordenados[i][0])
        i += 1

    return completados


def determinar_zona(pedidos: dict) -> dict:
    """
    PRECONDICIONES: pedidos es un diccionario que contiene los pedidos realizados con el formato
                       {numero de pedido (int): {'Fecha': fecha(DD/MM/AAAA) (str), 'Cliente': nombre (str),
                       'Ciudad': ciudad (str), 'Provincia': provincia (str), 'Artículos': {1334:{color(str):
                       {'Cantidad': cantidad(int), 'Descuento': descuento(int)},... },
                        568:{color(str): {'Cantidad': cantidad(int), 'Descuento': descuento(int)},... }}},...}
    POSTCONDICIONES: determina a que zona pertenece cada pedidos dependiendo de a que latitud está la ciudad,
                        retorna un diccionario con los pedidos separados por zona con el formato
                        {'Zona Norte':{'Pedidos': {...}} , 'Zona Centro':{'Pedidos': {...}},
                        'Zona Sur':{'Pedidos': {...}},'CABA':{'Pedidos': {...}}}
                        con {...} diccionarios con pedidos con el formato del diccionario 'pedidos'
    """

    zona_norte: dict = {}
    zona_centro: dict = {}
    zona_sur: dict = {}
    caba: dict = {}

    for pedido in pedidos:

        if pedidos[pedido]['Ciudad'] == 'CABA':

            caba[pedido] = pedidos[pedido]

        else:
            ciudad: str = pedidos[pedido]['Ciudad']
            locacion = geolocator.geocode(ciudad + ', ' + pedidos[pedido]['Provincia'] + ', Argentina')

            if abs(locacion.latitude) < 35:
                zona_norte[pedido] = pedidos[pedido]
            elif abs(locacion.latitude) < 40:
                zona_centro[pedido] = pedidos[pedido]
            else:
                zona_sur[pedido] = pedidos[pedido]

    pedidos_por_zona: dict = {'Zona Norte': {'Pedidos': zona_norte}, 'Zona Centro': {'Pedidos': zona_centro},
                              'Zona Sur': {'Pedidos': zona_sur}, 'CABA': {'Pedidos': caba}}

    return pedidos_por_zona


def determinar_recorrido(pedidos_por_zona: dict, zona: str) -> list:
    """
    PRECONDICIONES: pedidos_por_zona es un diccionario con los pedidos separado por zona con el formato
                       {'Zona Norte':{'Pedidos': {...}} , 'Zona Centro':{'Pedidos': {...}},
                       'Zona Sur':{'Pedidos': {...}},'CABA':{'Pedidos': {...}}}
                       con {...} diccionarios con pedidos con el formato
                       {numero de pedido (int): {'Fecha': fecha(DD/MM/AAAA) (str),
                        'Cliente': nombre (str), 'Ciudad': ciudad (str), 'Provincia': provincia (str),
                        'Artículos': {1334:{color(str): {'Cantidad': cantidad(int), 'Descuento': descuento(int)},... },
                        568:{color(str): {'Cantidad': cantidad(int), 'Descuento': descuento(int)},... }}},...}
                       zona es una de las siguientes cadenas 'Zona Norte', 'Zona Sur', 'Zona Centro'
    POSTCONDICIONES: determina el recorrido 'optimo' para pasar por todas las ciudades con pedidos de la zona
                        (en 'circulo' de izquierda a derecha (sentido horario para las zonas centro y sur,
                        antihorario para la zona norte))
                        y retorna una lista que tiene por elementos las ciudades en orden
    """

    pedidos_en_la_zona: dict = pedidos_por_zona[zona]['Pedidos']

    ciudades: list = []

    recorrido: list = []

    for pedido in pedidos_en_la_zona:

        ciudad: str = pedidos_en_la_zona[pedido]['Ciudad']
        if (ciudad + ', ' + pedidos_en_la_zona[pedido]['Provincia'] + ', Argentina') not in ciudades:
            ciudades.append(ciudad + ', ' + pedidos_en_la_zona[pedido]['Provincia'] + ', Argentina')

    if len(ciudades) > 0:
        latitudes: list = []
        longitudes: list = []

        for ciudad in ciudades:
            locacion = geolocator.geocode(ciudad)
            latitudes.append(locacion.latitude)
            longitudes.append(locacion.longitude)

        latitudes.sort()
        longitudes.sort()

        centro: int = ((longitudes[0]) + (longitudes[-1])) / 2

        oeste: list = []
        este: list = []

        for ciudad in ciudades:
            locacion = geolocator.geocode(ciudad)
            longitud: int = locacion.longitude

            if longitud < centro:
                oeste.append(ciudad)
            else:
                este.append(ciudad)

        latitudes_oeste: dict = {}
        for ciudad in oeste:
            locacion = geolocator.geocode(ciudad)
            latitud: int = locacion.latitude
            latitudes_oeste[ciudad] = latitud

        latitudes_este: dict = {}
        for ciudad in este:
            locacion = geolocator.geocode(ciudad)
            latitud: int = locacion.latitude
            latitudes_este[ciudad] = latitud

        if zona == 'Zona Norte':
            ordenadas_oeste = sorted(latitudes_oeste.items(), key=lambda x: x[1])
            ordenadas_este = sorted(latitudes_este.items(), key=lambda x: x[1], reverse=True)
        else:
            ordenadas_oeste = sorted(latitudes_oeste.items(), key=lambda x: x[1], reverse=True)
            ordenadas_este = sorted(latitudes_este.items(), key=lambda x: x[1])

        for ciudad in ordenadas_oeste:
            ciudad = list(ciudad[0].split(', '))
            recorrido.append(ciudad[0])
        for ciudad in ordenadas_este:
            ciudad = list(ciudad[0].split(', '))
            recorrido.append(ciudad[0])

    return recorrido


def distribucion_de_utilitarios(pedidos_por_zona: dict) -> dict:
    """
    PRECONDICIONES: pedidos_por_zona es un diccionario con los pedidos separado por zona con el formato {'Zona
                        Norte':{'Pedidos': {...}} , 'Zona Centro':{'Pedidos': {...}},'Zona Sur':{'Pedidos': {...}},
                        'CABA':{'Pedidos': {...}}} con {...} diccionarios con pedidos con el formato
                        {numero de pedido (int): {'Fecha': fecha(DD/MM/AAAA) (str), 'Cliente': nombre (str),
                         'Ciudad': ciudad (str), 'Provincia': provincia (str),
                         'Artículos': {1334:{color(str): {'Cantidad': cantidad(int), 'Descuento': descuento(int)},... },
                         568:{color(str): {'Cantidad': cantidad(int),'Descuento': descuento(int)},... }}},...}
    POISTCONDICIONES: determina el peso total de los pedidos de cada zona,
                        le asigna a cada zona (norte, sur, centro, CABA) un utilitario dependiendo del peso de los
                        pedidos a la zona con el mayor peso le asigna el utilitario que más peso soporta, etc. retorna
                         un diccionario con el utilitario y el peso por zona, con el formato
                          {'Zona Norte':{'Utilitario': numero de utilitario (str),'Peso': peso total de la
                        zona(int)} ,...}
    """

    peso_por_zona: dict = {}
    utilitarios: list = ['003', '001', '002', '004']

    utilitario_por_zona: dict = {'Zona Norte': {}, 'Zona Centro': {}, 'Zona Sur': {}, 'CABA': {}}

    for zona in pedidos_por_zona:

        peso: int = 0

        for pedido in pedidos_por_zona[zona]['Pedidos']:

            if 1334 in pedidos_por_zona[zona]['Pedidos'][pedido]['Artículos']:
                for color in pedidos_por_zona[zona]['Pedidos'][pedido]['Artículos'][1334]:
                    peso += int(pedidos_por_zona[zona]['Pedidos'][pedido]['Artículos'][1334][color]['Cantidad']) * 450
            if 568 in pedidos_por_zona[zona]['Pedidos'][pedido]['Artículos']:
                for color in pedidos_por_zona[zona]['Pedidos'][pedido]['Artículos'][568]:
                    peso += int(pedidos_por_zona[zona]['Pedidos'][pedido]['Artículos'][568][color]['Cantidad']) * 350

        peso_por_zona[zona] = peso

    ordenado = sorted(peso_por_zona.items(), key=lambda x: x[1])
    i: int = 0
    for zona in ordenado:
        utilitario_por_zona[zona[0]]['Utilitario'] = utilitarios[i]
        utilitario_por_zona[zona[0]]['Peso'] = zona[1]
        i += 1

    return utilitario_por_zona


def salida(pedidos: dict, stock: dict) -> None:
    """
    PRECONDICIONES: pedidos es un diccionario que contiene los pedidos realizados con el formato
                       {numero de pedido (int): {'Fecha': fecha(DD/MM/AAAA) (str), 'Cliente': nombre (str),
                        'Ciudad': ciudad (str), 'Provincia': provincia (str),
                        'Artículos': {1334:{color(str): {'Cantidad': cantidad(int), 'Descuento': descuento(int)},... },
                          568:{color(str): {'Cantidad': cantidad(int), 'Descuento': descuento(int)},... }}},...}
    POSTCONDICIONES: crea un archivo .txt listando por cada zona el utilitario que le corresponde, el peso total
                        de los pedidos en esa zona y las ciudades en el orden optimo de recorrido.
    """

    pedidos_comp_lista: list = pedidos_completos(pedidos, stock)
    pedidos_completados: dict = {}

    for pedido in pedidos_comp_lista:
        pedidos_completados[pedido] = pedidos[pedido]

    pedidos_por_zona: dict = determinar_zona(pedidos_completados)

    pedidos_z_u_p_r: dict = distribucion_de_utilitarios(pedidos_por_zona)

    for zona in pedidos_z_u_p_r:
        pedidos_z_u_p_r[zona]['Recorrido'] = ''
        if zona != 'CABA':
            recorrido: list = determinar_recorrido(pedidos_por_zona, zona)
            if len(recorrido) > 0:
                for i in range(len(recorrido) - 1):
                    pedidos_z_u_p_r[zona]['Recorrido'] += recorrido[i] + ', '
                pedidos_z_u_p_r[zona]['Recorrido'] += recorrido[-1]

    with open('salida.txt', 'w') as archivo_salida:
        for zona in pedidos_z_u_p_r:
            archivo_salida.writelines(f'''{zona}
Utilitario {pedidos_z_u_p_r[zona]['Utilitario']}
{pedidos_z_u_p_r[zona]['Peso'] / 1000}kg
{pedidos_z_u_p_r[zona]['Recorrido']}
''')


def mostrar_pedido(pedidos: dict, pedido: int) -> None:

    """
    PRECONDICIONES: pedidos es un diccionario que contiene los pedidos realizados con el formato
                      {numero de pedido (int): {'Fecha': fecha(DD/MM/AAAA) (str), 'Cliente': nombre (str),
                      'Ciudad': ciudad (str), 'Provincia': provincia (str), 'Artículos': {1334:{color(str):
                      {'Cantidad': cantidad(int), 'Descuento': descuento(int)},... },568:{color(str): {'Cantidad':
                      cantidad(int), 'Descuento': descuento(int)},... }}},...}
                      pedido es un entero
       POSTCONDICIONES: Si existe el pedido, lo muestra por pantalla.
    """

    if pedido in pedidos:
        print(pedido , ':')
        print('Fecha: ', pedidos[pedido]['Fecha'],', Cliente: ', pedidos[pedido]['Cliente'],', Ciudad: ', pedidos[pedido]['Ciudad'],', Provincia: ', pedidos[pedido]['Provincia'])
        print('Artículos: ')
        for articulo in pedidos[pedido]['Artículos']:
            print('Código: ', articulo)
            for color in pedidos[pedido]['Artículos'][articulo]:
                print('- Color: ', color)
                print('        - Cantidad: ', pedidos[pedido]['Artículos'][articulo][color]['Cantidad'])
                print('        - Descuento: ', pedidos[pedido]['Artículos'][articulo][color]['Descuento'])


def rosario(pedidos: dict, stock: dict) -> dict:
    """
        PRE: pedidos es un diccionario que contiene los pedidos realizados con el formato
                       {numero de pedido (int): {'Fecha': fecha(DD/MM/AAAA) (str), 'Cliente': nombre (str),
                       'Ciudad': ciudad (str), 'Provincia': provincia (str), 'Artículos': {1334:{color(str):
                       {'Cantidad': cantidad(int), 'Descuento': descuento(int)},... },568:{color(str): {'Cantidad':
                       cantidad(int), 'Descuento': descuento(int)},... }}},...},
                       stock es un diccionario con el stock disponible de vasos y botellas con el formato
                       {1334: {'Azul': cantidad(int), 'Rojo': cantidad(int),..}, 568:{'Azul': cantidad(int), 'Negro': cantidad(int)}}
       POST: Retorna un diccionario con los pedidos completos de la ciudad de Rosario
                        con el número de pedido (entero) como clave y el precio total del pedido como valor.
    """
    pedidos_comp_lista: list = pedidos_completos(pedidos, stock)

    pedidos_completados: dict = {}

    for pedido in pedidos_comp_lista:
        pedidos_completados[pedido] = pedidos[pedido]

    pedidos_rosario: dict = {}

    for pedido in pedidos_completados:

        if pedidos_completados[pedido]['Ciudad'] == 'Rosario' or pedidos_completados[pedido]['Ciudad'] == 'rosario':
            pedidos_rosario[pedido] = pedidos_completados[pedido]

    rosario_valores: dict = {}

    for pedido in pedidos_rosario:
        precio: int = 0

        if 1334 in pedidos_rosario[pedido]['Artículos']:
            for color in pedidos_rosario[pedido]['Artículos'][1334]:
                cantidad: int = pedidos_rosario[pedido]['Artículos'][1334][color]['Cantidad']
                descuento: int = pedidos_rosario[pedido]['Artículos'][1334][color]['Descuento']
                precio += (cantidad * 15) - ((cantidad * 15) * (descuento / 100))
        if 568 in pedidos_rosario[pedido]['Artículos']:
            for color in pedidos_rosario[pedido]['Artículos'][568]:
                cantidad: int = pedidos_rosario[pedido]['Artículos'][568][color]['Cantidad']
                descuento: int = pedidos_rosario[pedido]['Artículos'][568][color]['Descuento']
                precio += (cantidad * 8) - ((cantidad * 8) * (descuento / 100))

        rosario_valores[pedido] = precio

    return rosario_valores


def mas_pedido(pedidos: dict, stock: dict) -> tuple:
    """
    PRE: pedidos es un diccionario que contiene los pedidos realizados con el formato
                       {numero de pedido (int): {'Fecha': fecha(DD/MM/AAAA) (str), 'Cliente': nombre (str),
                       'Ciudad': ciudad (str), 'Provincia': provincia (str), 'Artículos': {1334:{color(str):
                       {'Cantidad': cantidad(int), 'Descuento': descuento(int)},... },568:{color(str): {'Cantidad':
                       cantidad(int), 'Descuento': descuento(int)},... }}},...},
                       stock es un diccionario con el stock disponible de vasos y botellas con el formato
                       {1334: {'Azul': cantidad(int), 'Rojo': cantidad(int),..},
                        568:{'Azul': cantidad(int), 'Negro': cantidad(int)}}
    POST: Retorna una tupla con el nombre del artículo más pedido (tipo y color), la cantidad de unidades
                        de este que se pidieron(int) y la cantidad de unidades que se entregaron(int):
                        nombre, pedidos, entregados

    """
    pedidos_comp_lista: list = pedidos_completos(pedidos, stock)
    pedidos_completados: dict = {}

    for pedido in pedidos_comp_lista:
        pedidos_completados[pedido] = pedidos[pedido]

    cantidades: dict = {}

    for pedido in pedidos:
        for articulo in pedidos[pedido]['Artículos']:
            for color in pedidos[pedido]['Artículos'][articulo]:
                if (str(articulo) + ' ' + color) in cantidades:
                    cantidades[str(articulo) + ' ' + color] = cantidades[str(articulo) + ' ' + color] + \
                                                              pedidos[pedido]['Artículos'][articulo][color]['Cantidad']
                else:
                    cantidades[str(articulo) + ' ' + color] = pedidos[pedido]['Artículos'][articulo][color]['Cantidad']

    ordenados = sorted(cantidades.items(), key=lambda x: x[1], reverse=True)
    cantidad_pedidos: int = ordenados[0][1]
    articulo = ordenados[0][0]
    articulo = list(articulo.split(' '))
    if articulo[0] == '1334':
        nombre = 'Botella en ' + articulo[1]
    else:
        nombre = 'Vaso en ' + articulo[1]

    cantidad_entregados: int = 0
    for pedido in pedidos_completados:
        if int(articulo[0]) in pedidos_completados[pedido]['Artículos']:
            if articulo[1] in pedidos_completados[pedido]['Artículos'][int(articulo[0])]:
                cantidad_entregados += pedidos_completados[pedido]['Artículos'][int(articulo[0])][articulo[1]][
                    'Cantidad']

    return nombre, cantidad_pedidos, cantidad_entregados


def main():
    pedidos: dict = lectura_pedidos('pedidos.csv')

    stock: dict = {}
    try:
        cargar_yolo()
    except:
        print('No se encontraron todos los archivos de AI')
    else:
        stock = clasificar_stock()

    opciones: list = ["ABM", "Calcular Recorrido", "Proceso Pedidos", "Listado Pedidos Completados",
                      "Pedidos Valorizados Ciudad de Rosario", "Articulo mas pedido", "Archivos", "Salir"]
    opcion: str = ''

    while opcion != '8':

        print("Menu: ")
        for indice in range(len(opciones)):
            print(indice + 1, "- ", opciones[indice])

        opcion = input("")

        if opcion == '1':

            pedidos = abm(pedidos)


        elif opcion == '2':

            if len(pedidos) != 0:
                print('Elija una de las siguientes zonas para calcular su recorrido ')
                zona_elegida: str = input('Zona Norte, Zona Centro, Zona Sur: ')
                while zona_elegida != 'Zona Norte' and zona_elegida != 'Zona Sur' and zona_elegida != 'Zona Centro':
                    print('INGRESO INVÁLIDO')
                    print('Elija una de las siguientes zonas para calcular su recorrido ')
                    zona_elegida = input('Zona Norte, Zona Centro, Zona Sur: ')

                pedidos_comp_lista: list = pedidos_completos(pedidos, stock)
                pedidos_completados: dict = {}

                for pedido in pedidos_comp_lista:
                    pedidos_completados[pedido] = pedidos[pedido]

                pedidos_zonas: dict = determinar_zona(pedidos_completados)

                recorrido_zona: list = determinar_recorrido(pedidos_zonas, zona_elegida)
                if len(recorrido_zona) > 0:
                    print(f'El recorrido de la {zona_elegida} es: ')
                    for i in range(len(recorrido_zona) - 1):
                        print(recorrido_zona[i], end=', ')
                    print(recorrido_zona[-1])
                else:
                    print('No hay pedidos en esta zona.')
            else:

                print('El diccionario esta vacio')

        elif opcion == '3':

            if len(pedidos) != 0:

                salida(pedidos, stock)
                print("Se creo un archivo 'salida.txt' con la información de los pedidos")

            else:

                print('el diccionario esta vacio')

        elif opcion == '4':

            if len(pedidos) != 0:

                print('Los pedidos que se pudieron completar son:')
                completados: list = pedidos_completos(pedidos, stock)
                for pedido in completados:
                    mostrar_pedido(pedidos, pedido)

            else:

                print('el diccionario esta vacio')

        elif opcion == '5':

            if len(pedidos) != 0:

                total: int = 0
                print('Los siguientes pedidos fueron a la ciudad de rosario:')
                pedidos_rosario: dict = rosario(pedidos, stock)
                for pedido in pedidos_rosario:
                    print(f'Pedido {pedido} -> {pedidos_rosario[pedido]} dólares')
                    total += pedidos_rosario[pedido]
                print(f'Total: {total} dólares')

            else:

                print('el diccionario esta vacio')

        elif opcion == '6':

            if len(pedidos) != 0:

                articulo, cant_pedidos, cant_entregados = mas_pedido(pedidos, stock)

                print(f'El artículo más pedido fue {articulo}')
                print(f'Se pidieron {cant_pedidos} unidades y de ellas se entregaron {cant_entregados}')

            else:

                print('el diccionario esta vacio')

        elif opcion == '7':

            if len(pedidos) != 0:

                crear_archivos(stock)
                print('Se crearon dos archivos de texto, botellas.txt y vasos.txt,')
                print('con la cantidad de botellas y vasos procesados por color, respectivamente.')

            else:

                print('el diccionario esta vacio')

        elif opcion == '8':

            print("Cordial Despedida")

        else:

            print("Las opciones deben ser entre 1 y 8")

main()