from math import pi
from math import isclose
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import copy
import numpy as np
from math import sqrt

limit = 2 * pi
clockwiseSpin = False

def column(matrix, i):
	return [row[i] for row in matrix]

def calculateDelta(start, end, spinRightOrientation: bool = True):
	if isclose(end, start, abs_tol=0.00001):
		return 0

	end = (limit + end) % limit
	start = (limit + start) % limit

	ans = abs(end - start)
	if spinRightOrientation:
		if start < end:
			return limit - ans
		return ans
	if start > end:
		return limit - ans
	return ans

def getDeltaAngle(angInicial, angFinal):
	global clockwiseSpin

	delta = calculateDelta(angInicial, angFinal, clockwiseSpin)

	# Hack: spin orientation change
	if delta > pi:
		delta = (2 * pi) - delta
		clockwiseSpin = not clockwiseSpin

	return delta

def calcVelocidadeRoda(angInicial, angFinal, deltaTime):
	global clockwiseSpin
	deltaA = getDeltaAngle(angInicial, angFinal)
	speed = deltaA / deltaTime

	if clockwiseSpin:
		return -speed
	return speed

def distPontos(p1, p2):
	return sqrt(((p2[0]-p1[0])**2) + ((p2[1]-p1[1])**2))

def clusterPoints(nuvemPontos, dist, vizinhaca):
	pontosExcluidos = []
	centroidsCluters = []
	for p in nuvemPontos:
		if p not in pontosExcluidos:
			vizinhos = 0
			for pn in nuvemPontos:
				if distPontos(p, pn) < dist:
					pontosExcluidos.append(pn)
					vizinhos += 1
			if vizinhos > vizinhaca:
				#print(vizinhos)
				centroidsCluters.append(p)
	return centroidsCluters


def pontoMaisProx(ponto, listaPontos):
	menorDist = 99999999
	menorP = listaPontos[0]
	for p in listaPontos:
		d = distPontos(ponto, p)
		if d < menorDist:
			menorDist = d
			menorP = p
	return menorP



def incrementalFindLines(nuvemPontos, tresholdError):
	novosPontos = copy.copy(nuvemPontos)
	linhas = []
	pontosX = None
	pontosY = None
	i = 0
	while(True):
		#print("aaaaaaaaaa")
		if len(novosPontos) < 2:
			break
		p1 = novosPontos[0]
		novosPontos.remove(p1)
		p2 = pontoMaisProx(p1, novosPontos)
		novosPontos.remove(p2)
		pontosX = np.array([p1[0], p2[0]])
		pontosY = np.array([p1[1], p2[1]])
		pInicial = p1
		pFinal = p2
		while(True):
			if len(novosPontos) < 1:
				break
			#print("p=",nuvemPontos[i])
			pAdd = pontoMaisProx(pFinal, novosPontos)
			pontosX = np.append(pontosX,pAdd[0])
			pontosY = np.append(pontosY,pAdd[1])
			regr = linear_model.LinearRegression()
			mX = np.matrix(pontosX).T
			mY = np.matrix(pontosY).T
			#print("pontosX=",mX.T)
			#print("pontosY=",mY.T)
			regr.fit(mX, mY)
			predY = regr.predict(mX)
			error = mean_squared_error(mY, predY)
			#print("erro/coef/linear", error,"/", regr.coef_, regr.intercept_)
			if(error > tresholdError):
				break
			else:
				pFinal = pAdd
				novosPontos.remove(pAdd)
		linhas.append([pInicial, pFinal])

	return linhas



