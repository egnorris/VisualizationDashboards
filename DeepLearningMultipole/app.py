#import sys, os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"


#import tensorflow as tf
import json
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.io import savemat


import imageio.v2 as imageio

import matplotlib
font = {'family' : 'sans serif',
        'size'   : 28}
from matplotlib.colors import LinearSegmentedColormap

from dash import Dash, html, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px

matplotlib.rc('font', **font)
def FiniteDifference(f,coord, order, params):
    N_theta, N_phi = f.shape
    result = f * 0
    step_size = params["step"]
    d_theta = step_size[0]
    d_phi = step_size[1]
    if coord == "theta" and order == 1:
        for i_theta in range(1,N_theta-1):
            for i_phi in range(1,N_phi-1):
                result[i_theta,i_phi] = (f[i_theta+1,i_phi] - f[i_theta-1,i_phi])/(2*d_theta)
        return result
    elif coord == "phi" and order == 1:
        for i_theta in range(1,N_theta-1):
            for i_phi in range(1,N_phi-1):
                result[i_theta,i_phi] = (f[i_theta,i_phi+1] - f[i_theta,i_phi-1])/(2*d_phi)
        return result
    elif coord == "theta" and order == 2:
        for i_theta in range(1,N_theta-1):
            for i_phi in range(1,N_phi-1):
                result[i_theta,i_phi] = (f[i_theta+1,i_phi] - 2*f[i_theta,i_phi] + f[i_theta-1,i_phi])/(d_theta*d_theta)
        return result
    elif coord == "phi" and order == 2:
        for i_theta in range(1,N_theta-1):
            for i_phi in range(1,N_phi-1):
                result[i_theta,i_phi] = (f[i_theta,i_phi+1] - 2*f[i_theta,i_phi] + f[i_theta,i_phi-1])/(d_phi*d_phi)
        return result

def hankel(params,derivative,conj):
    k = params["k"]
    r = params["r"]
    l = params["l"]
    if conj == True:
        i = -1j
    else:
        i = 1j
    if derivative == False:
        return (-i)**(l+1)*np.exp(i*k*r) / (k*r)
    elif derivative == True:
        return (-i)**(l)*np.exp(i*k*r)*(r+i) / (k*r*r)

def GetSphericalComponent(params, SphHarmDict, FieldType, Component):
    hbar = 1.054571817*10**(-34)
    const = hbar / np.sqrt(params["l"]*(params["l"]+1))

    if Component == "theta":

        if FieldType == "electric":

            Field = params["aE"]*1j*hankel(params,True,False)*SphHarmDict["D"]["1"]["theta"]/params["k"]
            Field = Field + params["aH"]*hankel(params,False,True)*SphHarmDict["D"]["1"]["phi"]/np.sin(params["theta"])
            Field = Field * -1j * const
            return Field
        
        elif FieldType == "magnetic":

            Field = params["aH"]* -1j *hankel(params,True,True)*SphHarmDict["D"]["1"]["theta"]/params["k"]
            Field = Field + params["aE"]*hankel(params,False,False)*SphHarmDict["D"]["1"]["phi"]/np.sin(params["theta"])
            Field = Field * 1j * const
            return Field

    elif Component == "phi":

        if FieldType == "electric":

            Field = params["aE"]*1j*hankel(params,True,False)*SphHarmDict["D"]["1"]["phi"]/(params["k"] * np.sin(params["theta"]))
            Field = Field + params["aH"]*hankel(params, False, True)*SphHarmDict["D"]["1"]["theta"]
            Field = Field * -1j * const
            return Field
        
        elif FieldType == "magnetic":

            Field = params["aH"]*1j*hankel(params,True,True)*SphHarmDict["D"]["1"]["phi"]/(params["k"] * np.sin(params["theta"]))
            Field = Field + params["aE"]*hankel(params, False, False)*SphHarmDict["D"]["1"]["theta"]
            Field = Field * -1j * const
            return Field

    elif Component == "r":

        if FieldType == "electric":
            Field = params["aE"]*1j*hankel(params,False,False)/(params["k"]*params["r"])
            Field = Field * (SphHarmDict["D"]["2"]["theta"] + SphHarmDict["D"]["2"]["phi"]/(np.sin(params["theta"])**2))
            Field = -1j * const * Field
            return Field

        elif FieldType == "magnetic":
            Field = params["aE"]*1j*hankel(params,False,True)/(params["k"]*params["r"])
            Field = Field * (SphHarmDict["D"]["2"]["theta"] + SphHarmDict["D"]["2"]["phi"]/(np.sin(params["theta"])**2))
            Field = 1j * const * Field
            return Field
def PercentDifference(A,B):
        return 100*abs(A-B)/((A+B)/2)

def cartesian_projection(F,params):
    PHI = params["phi"]
    THETA = params["theta"]
    F_r , F_theta, F_phi = F

    x_hat_r = np.sin(THETA)*np.cos(PHI); x_hat_theta = np.cos(THETA)*np.cos(PHI); x_hat_phi = -np.sin(PHI)
    z_hat_r = np.sin(THETA)*np.sin(PHI); z_hat_theta = np.cos(THETA)*np.sin(PHI); z_hat_phi = np.cos(PHI)
    y_hat_r = np.cos(THETA); y_hat_theta = -np.sin(THETA); y_hat_phi = 0

    F_x = (x_hat_r * F_r) + (x_hat_theta * F_theta) + (x_hat_phi * F_phi)
    F_y = (y_hat_r * F_r) + (y_hat_theta * F_theta) + (y_hat_phi * F_phi)
    F_z = (z_hat_r * F_r) + (z_hat_theta * F_theta) + (z_hat_phi * F_phi)
    return [F_x, F_y, F_z]

def GetFieldComponent(params):
    SphHarm = sp.sph_harm(params["m"],params["l"],params["phi"],params["theta"])
    #Calculate Derivatives of Spherical Harmonic, Y
    SphHarmFirstDerivPhi   = FiniteDifference(SphHarm,'phi', 1, params)
    SphHarmFirstDerivTheta  = FiniteDifference(SphHarm,'theta', 1, params)
    SphHarmSecondDerivPhi   = FiniteDifference(SphHarm,'phi', 2, params)
    SphHarmSecondDerivTheta = FiniteDifference(SphHarm,'theta', 2, params)
    FirstDerivatives = {"phi":SphHarmFirstDerivPhi, "theta":SphHarmFirstDerivTheta}
    SecondDerivatives = {"phi":SphHarmSecondDerivPhi, "theta":SphHarmSecondDerivTheta}
    DerivativeDict = {"1": FirstDerivatives, "2": SecondDerivatives}
    SphHarmDict = {"Y": SphHarm, "D": DerivativeDict}

    ElecFieldR     = GetSphericalComponent(params, SphHarmDict, "electric", "r")
    ElecFieldTheta = GetSphericalComponent(params, SphHarmDict, "electric", "theta")
    ElecFieldPhi   = GetSphericalComponent(params, SphHarmDict, "electric", "phi")
    MagnFieldR     = GetSphericalComponent(params, SphHarmDict, "magnetic", "r")
    MagnFieldTheta = GetSphericalComponent(params, SphHarmDict, "magnetic", "theta")
    MagnFieldPhi   = GetSphericalComponent(params, SphHarmDict, "magnetic", "phi")

    ElecFieldCart = cartesian_projection([ElecFieldR, ElecFieldTheta, ElecFieldPhi], params)
    MagnFeildCart = cartesian_projection([MagnFieldR, MagnFieldTheta, MagnFieldPhi], params)

    return [ElecFieldCart, MagnFeildCart]

def magnitude(z):
  x = np.real(z)
  y = np.imag(z)
  return np.sqrt(x**2 + y**2)

def phase(z):
  x = np.real(z)
  y = np.imag(z)
  return np.arctan2(y,x)


def GetField(aE,aH, wavelength, d_theta, d_phi):
    theta = np.arange(0,np.pi, d_theta)
    phi = np.arange(0,2*np.pi, d_phi)
    dims = [len(theta), len(phi)]
    PHI,THETA = np.meshgrid(phi, theta)
    wavelength = wavelength*10**(-9)
    params = {
        "l":1,
        "m":1,
        "phi":PHI,
        "theta":THETA,
        "r":300*10**(-6),
        "k":2*np.pi / wavelength,
        "step": [d_theta, d_phi],
        "aE":aE[0],
        "aH":aH[0]}
    E0, H0 = GetFieldComponent(params)
    params["l"] = 2
    params["aE"] = aE[1]; params["aH"] = aH[1]
    E1, H1 = GetFieldComponent(params)
    params["m"] = 2
    params["aE"] = aE[2]; params["aH"] = aH[2]
    E2, H2 = GetFieldComponent(params)
    E0x, E0y, E0z = E0
    H0x, H0y, H0z = H0
    E1x, E1y, E1z = E1
    H1x, H1y, H1z = H1
    E2x, E2y, E2z = E2
    H2x, H2y, H2z = H2
    E = [E0x + E1x + E2x, E0y + E1y + E2y, E0z + E1z + E2z]
    H = [H0x + H1x + H2x, H0y + H1y + H2y, H0z + H1z + H2z]
    return [E, H]


def GetFieldSubplot(Field, FigShape, Location, Label, Axes = [False, False]):
    ax = plt.subplot2grid(shape = FigShape, loc=Location, colspan=1)
    ax.set_title(Label)
    ax.get_xaxis().set_visible(Axes[0])
    ax.get_yaxis().set_visible(Axes[1])
    ax.set_ylabel("θ")
    ax.set_xlabel("ɸ")
    p = ax.imshow(Field,extent=[0,2*np.pi,0,np.pi])
    return ax
def GetErrorSubplot(Field, FigShape, Location, Label, Axes = [False, False]):
    #cmap = LinearSegmentedColormap.from_list("", ["blue","red"], N = 2)
    ax = plt.subplot2grid(shape = FigShape, loc=Location, colspan=1)
    ax.set_title(Label)
    ax.get_xaxis().set_visible(Axes[0])
    ax.get_yaxis().set_visible(Axes[1])
    ax.set_ylabel("θ")
    ax.set_xlabel("ɸ")
    p = ax.imshow(Field,extent=[0,2*np.pi,0,np.pi], cmap = "binary")
    p.set_clim(0,100)
    return ax


def PlotComponent(Component, Wavelengths, Label, Color, WavelengthIDX, MarkerLabel):
    wl = round(Wavelengths[WavelengthIDX] * 1E9)
    ax = plt.plot(Wavelengths * 1E9, Component, label=Label, color=Color, linewidth = 3)
    if MarkerLabel == True:
        ax = plt.plot(Wavelengths[WavelengthIDX] * 1E9, Component[WavelengthIDX], '.', color = "black", markersize = 20, label = "λ: {} nm".format(wl))
        
    else:
        ax = plt.plot(Wavelengths[WavelengthIDX] * 1E9, Component[WavelengthIDX], '.', color = "black", markersize = 20)
    return ax

def GetComponentSubplot(Component, Wavelengths, WavelengthIDX, FigShape, Location, FieldType):
    ax = plt.subplot2grid(shape = FigShape, loc=Location, colspan=2)
    ax.plot()
    ax.set_xlabel("λ [nm]")
    ax = PlotComponent(Component[0], Wavelengths, f"$a_{{1,1}}^{FieldType[0]}$ SIM", "darkred", WavelengthIDX, False)
    ax = PlotComponent(Component[3], Wavelengths, f"$a_{{1,1}}^{FieldType[0]}$ CNN", "red", WavelengthIDX, False)
    ax = PlotComponent(Component[1], Wavelengths, f"$a_{{2,1}}^{FieldType[0]}$ SIM", "navy", WavelengthIDX, False)
    ax = PlotComponent(Component[4], Wavelengths, f"$a_{{2,1}}^{FieldType[0]}$ CNN", "cornflowerblue", WavelengthIDX, False)
    ax = PlotComponent(Component[2], Wavelengths, f"$a_{{2,2}}^{FieldType[0]}$ SIM", "darkgreen", WavelengthIDX, False)
    ax = PlotComponent(Component[5], Wavelengths, f"$a_{{2,2}}^{FieldType[0]}$ CNN", "springgreen", WavelengthIDX, True)
    
    plt.legend(bbox_to_anchor = (1.1, -0.2), ncol = 7)    
    return ax


def GetShapeSubplot(Shape, FigShape, Location):
    ax = plt.subplot2grid(shape = FigShape, loc=Location, colspan=1)
    ax.imshow(Shape)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax

def PlotField(Field, Components, FieldType, Representation, Shape, ShapeIDX, Wavelengths, WavelengthIDX, SavePath, FigSize):
    FigShape = (3,3)
    if Representation == "Magnitude":
        plt.set_cmap(plt.get_cmap('inferno'))
        enclosure = ["|", "|"]
        for i in range(6):
            Field[i] = magnitude(Field[i])

    elif Representation == "Phase":
        plt.set_cmap(plt.get_cmap('hsv'))
        enclosure = ["arg(", ")"]
        for i in range(6):
            Field[i] = phase(Field[i])

    SimEx, SimEy, SimEz, CnnEx, CnnEy, CnnEz = Field

    fig = plt.figure()
    fig.set_figheight(FigSize[0])
    fig.set_figwidth(FigSize[1])
    fig.suptitle(f"Comparison of Predicted and Simulated Multipole Components in {FieldType} Far-Field Generation", fontsize=40)
    if FieldType == "Magnetic":
        FieldType = "Hagnetic"
    ax1 = GetFieldSubplot(SimEx, FigShape, (0,0), f"{enclosure[0]}${FieldType[0]}_{{x}}${enclosure[1]} SIM", Axes = [False, True])
    ax2 = GetFieldSubplot(SimEy, FigShape, (0,1), f"{enclosure[0]}${FieldType[0]}_{{y}}${enclosure[1]} SIM")
    ax3 = GetFieldSubplot(SimEz, FigShape, (0,2), f"{enclosure[0]}${FieldType[0]}_{{z}}${enclosure[1]} SIM")
    ax4 = GetFieldSubplot(CnnEx, FigShape, (1,0), f"{enclosure[0]}${FieldType[0]}_{{x}}${enclosure[1]} CNN", Axes = [True, True])
    ax5 = GetFieldSubplot(CnnEy, FigShape, (1,1), f"{enclosure[0]}${FieldType[0]}_{{y}}${enclosure[1]} CNN", Axes = [True, False])
    ax6 = GetFieldSubplot(CnnEz, FigShape, (1,2), f"{enclosure[0]}${FieldType[0]}_{{z}}${enclosure[1]} CNN", Axes = [True, False])
    ax7 = GetComponentSubplot(Components, Wavelengths, WavelengthIDX, FigShape, (2,0), FieldType)
    plt.set_cmap(plt.get_cmap('inferno'))
    ax8 = GetShapeSubplot(profiles[ShapeIDX], FigShape, (2,2))
    if FieldType == "Hagnetic":
        FieldType = "Magnetic"
    plt.savefig(f"{SavePath}{FieldType}{Representation}-{round(Wavelengths[WavelengthIDX]*1E9)}nm.png")
    plt.close()

def makemovie(FieldType, PlotType, ShapeIDX, Wavelengths, WavelengthRange=[0, 101, 1]):
    images = []
    for i in np.arange(WavelengthRange[0],WavelengthRange[1],WavelengthRange[2]):
            wl = round(Wavelengths[i]*1E9)
            images.append(imageio.imread(f"/media/work/evan/MultipoleFieldImageData/temp/{FieldType}{PlotType}-{wl}nm.png"))
    imageio.mimsave(f"/media/work/evan/MultipoleFieldImageData/movie/{FieldType}{PlotType}-Shape{ShapeIDX}.gif", images, format = 'GIF', loop = 0, fps = 10)

def PlotError(Field, Components, FieldType, Representation, Shape, ShapeIDX, Wavelengths, WavelengthIDX, SavePath, FigSize):
    FigShape = (3,3)

    SimEx, SimEy, SimEz, CnnEx, CnnEy, CnnEz = Field

    

    fig = plt.figure()
    fig.set_figheight(FigSize[0])
    fig.set_figwidth(FigSize[1])
    fig.suptitle(f"Pixel Difference Between FDTD and CNN Prediction for {FieldType} Far-Field", fontsize=40)
    if FieldType == "Magnetic":
        FieldType = "Hagnetic"
    ax1 = GetErrorSubplot(PercentDifference(magnitude(SimEx),magnitude(CnnEx)), FigShape, (0,0), f"Magnitude ${FieldType[0]}_{{x}}$ Difference")
    ax2 = GetErrorSubplot(PercentDifference(magnitude(SimEy),magnitude(CnnEy)), FigShape, (0,1), f"Magnitude ${FieldType[0]}_{{y}}$ Difference")
    ax3 = GetErrorSubplot(PercentDifference(magnitude(SimEz),magnitude(CnnEz)), FigShape, (0,2), f"Magnitude ${FieldType[0]}_{{z}}$ Difference")
    ax4 = GetErrorSubplot(PercentDifference(phase(SimEx),phase(CnnEx)), FigShape, (1,0), f"Phase ${FieldType[0]}_{{x}}$ Difference")
    ax5 = GetErrorSubplot(PercentDifference(phase(SimEy),phase(CnnEy)), FigShape, (1,1), f"Phase ${FieldType[0]}_{{y}}$ Difference")
    ax6 = GetErrorSubplot(PercentDifference(phase(SimEz),phase(CnnEz)), FigShape, (1,2), f"Phase ${FieldType[0]}_{{z}}$ Difference")
    plt.set_cmap(plt.get_cmap('inferno'))
    ax7 = GetComponentSubplot(Components, Wavelengths, WavelengthIDX, FigShape, (2,0), FieldType)
    ax8 = GetShapeSubplot(profiles[ShapeIDX], FigShape, (2,2))
    if FieldType == "Hagnetic":
        FieldType = "Magnetic"
    plt.savefig(f"{SavePath}{FieldType}Error-{round(Wavelengths[WavelengthIDX]*1E9)}nm.png")
    #plt.show()
    plt.close()
"""
    if FieldType == "Electric" and PlotType == "Magnitude":
        A = data[0]
        colormap = 'jet'
"""
def DataSelectionTruth(Data, FieldType, PlotType, Coord):
    if FieldType == "Electric" and PlotType == "Magnitude":
        if Coord == "X":
            return Data[0]
        elif Coord == "Y":
            return Data[1]
        elif Coord == "Z":
            return Data[2]
    if FieldType == "Electric" and PlotType == "Phase":
        if Coord == "X":
            return Data[12]
        elif Coord == "Y":
            return Data[13]
        elif Coord == "Z":
            return Data[14]
    if FieldType == "Magnetic" and PlotType == "Magnitude":
        if Coord == "X":
            return Data[6]
        elif Coord == "Y":
            return Data[7]
        elif Coord == "Z":
            return Data[8]
    if FieldType == "Magnetic" and PlotType == "phase":
        if Coord == "X":
            return Data[18]
        elif Coord == "Y":
            return Data[19]
        elif Coord == "Z":
            return Data[20]
        
def DataSelectionPrediction(Data, FieldType, PlotType, Coord):
    if FieldType == "Electric" and PlotType == "Magnitude":
        if Coord == "X":
            return Data[3]
        elif Coord == "Y":
            return Data[4]
        elif Coord == "Z":
            return Data[5]
    if FieldType == "Electric" and PlotType == "Phase":
        if Coord == "X":
            return Data[15]
        elif Coord == "Y":
            return Data[16]
        elif Coord == "Z":
            return Data[17]
    if FieldType == "Magnetic" and PlotType == "Magnitude":
        if Coord == "X":
            return Data[9]
        elif Coord == "Y":
            return Data[10]
        elif Coord == "Z":
            return Data[11]
    if FieldType == "Magnetic" and PlotType == "phase":
        if Coord == "X":
            return Data[21]
        elif Coord == "Y":
            return Data[22]
        elif Coord == "Z":
            return Data[23]


with open('DeepLearningMultipoleData.pkl', 'rb') as f:
    DataSet = pkl.load(f)


app = Dash(__name__)

app.layout = html.Div(children=[
    html.H1(
        children="Deep Learning Multipole Expansion Visualization",
        style={'textAlign': 'center'}),
    html.Div([
        dcc.Graph(
            id='Multipole-Spectrum-Graph'
        )], style = {'width': '70%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(
            id='Multipole-Profile-Graph'
        )], style = {'width': '30%','display': 'inline-block'}),
    html.Div([
        dcc.Slider(
            300,
            800,
            5,
            value = 300,
            id='Wavelength-Slider',
            marks=None,
            tooltip={
                "placement": "top",
                "always_visible": True,
                "template": "{value} nm"
            }
        )
    ]),
    html.Div([
        dcc.Slider(
            0,
            len(DataSet["Spectrum Data"])-1,
            1,
            value = 100,
            id='Profile-Slider',
            marks=None,
            tooltip={
                "placement": "bottom",
                "always_visible": True
            }
        )
    ]),
    html.Div([
            dcc.RadioItems(
                ['Electric', 'Magnetic'],
                'Electric',
                id='Field Type',
                inline=True)
        ], style={'width': '100%', 'text-align': 'center', 'display': 'inline-block', 'font-size': 32}),
    html.Div([html.Br()]),
    html.Div([
            dcc.RadioItems(
                ['Magnitude', 'Phase'],
                'Magnitude',
                id='Plot Type',
                inline=True)
        ], style={'width': '100%', 'text-align': 'center', 'display': 'inline-block', 'font-size': 32}),
    html.Div([html.Br()]),
    html.Div([
        dcc.Graph(
            id='Ex-Truth-Graph'
        )], style = {'width': '33%','display': 'inline-block'}),
    html.Div([
        dcc.Graph(
            id='Ey-Truth-Graph'
        )], style = {'width': '33%','display': 'inline-block'}),
    html.Div([
        dcc.Graph(
            id='Ez-Truth-Graph'
        )], style = {'width': '33%','display': 'inline-block'}),
    html.Div([html.Br()]),
    html.Div([
        dcc.Graph(
            id='Ex-CNN-Graph'
        )], style = {'width': '33%','display': 'inline-block'}),
    html.Div([
        dcc.Graph(
            id='Ey-CNN-Graph'
        )], style = {'width': '33%','display': 'inline-block'}),
    html.Div([
        dcc.Graph(
            id='Ez-CNN-Graph'
        )], style = {'width': '33%','display': 'inline-block'}),
    dcc.Store(id='FieldData')
])

@callback(
    Output('Multipole-Spectrum-Graph', 'figure'),
    Input('Wavelength-Slider', 'value'),
    Input('Profile-Slider', 'value'),
)
def update_figure(value1, value2):
    df = pd.DataFrame(DataSet["Spectrum Data"][value2])
    df = df.set_index("Wavelength [nm]")
    fig = px.line(df, labels = {"value":"Component"})
    fig.add_scatter(x=list(map(int, (np.ones(12) * value1).tolist())),
    y=df.loc[value1],
    marker_color='black',
    name=f"{value1} nm")
    return fig

@callback(
    Output('FieldData', 'data'),
    Input('Profile-Slider', 'value'),
    Input('Wavelength-Slider', 'value'),
)
def calculate_field(ShapeIDX, Wavelength):
    df = pd.DataFrame(DataSet["Spectrum Data"][ShapeIDX])
    df = df.set_index("Wavelength [nm]")
    components = df.loc[Wavelength].to_numpy().tolist()
    wl = Wavelength*1E-9
    aE = [components[0], components[2], components[4]]
    aH = [components[6], components[8], components[10]]
    dTheta = 0.05
    dPhi = 0.05 
    E0, H0 = GetField(aE,aH, wl, dTheta, dPhi)
    Ex0, Ey0, Ez0 = E0
    Hx0, Hy0, Hz0 = H0
    aE = [components[1], components[3], components[5]]
    aH = [components[7], components[9], components[11]]
    E1, H1 = GetField(aE,aH, wl, dTheta, dPhi)
    Ex1, Ey1, Ez1 = E1
    Hx1, Hy1, Hz1 = H1
    return [magnitude(Ex0), magnitude(Ey0), magnitude(Ez0),
            magnitude(Ex1), magnitude(Ey1), magnitude(Ez1),
            magnitude(Hx0), magnitude(Hy0), magnitude(Hz0),
            magnitude(Hx1), magnitude(Hy1), magnitude(Hz1),
            phase(Ex0), phase(Ey0), phase(Ez0),
            phase(Ex1), phase(Ey1), phase(Ez1),
            phase(Hx0), phase(Hy0), phase(Hz0),
            phase(Hx1), phase(Hy1), phase(Hz1)]


@callback(
    Output('Multipole-Profile-Graph', 'figure'),
    Input('Profile-Slider', 'value'),
)
def update_figure(value):
    fig = px.imshow(DataSet["Geometry Data"][value])
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_yaxes(showticklabels=False)
    return fig

@callback(
    Output('Ex-Truth-Graph', 'figure'),
    Input('FieldData', 'data'),
    Input('Field Type', 'value'),
    Input('Plot Type', 'value')
)
def update_figure(data, FieldType, PlotType):
    A = DataSelectionTruth(data, FieldType, PlotType, "X")
    if PlotType == "Phase":
        colormap = "edge"
    elif PlotType == "Magnitude":
        colormap = "jet"
    if FieldType == "Magnetic":
        FieldType = "Hagnetic"
    fig = px.imshow(A, color_continuous_scale=colormap, labels = {"y": "FDTD"})
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(title_text=f"{FieldType[0]}<sub>x</sub> {PlotType}", title_x=0.5)
    fig.update_layout(font=dict(size=32, family="American Typewriter", color="Black"))
    fig.update_layout(coloraxis_showscale=False)
    fig.update_yaxes(showticklabels=False)
    return fig

@callback(
    Output('Ey-Truth-Graph', 'figure'),
    Input('FieldData', 'data'),
    Input('Field Type', 'value'),
    Input('Plot Type', 'value')
)
def update_figure(data, FieldType, PlotType):
    A = DataSelectionTruth(data, FieldType, PlotType, "Y")
    if PlotType == "Phase":
        colormap = "edge"
    elif PlotType == "Magnitude":
        colormap = "jet"
    if FieldType == "Magnetic":
        FieldType = "Hagnetic"
    fig = px.imshow(A, color_continuous_scale=colormap)
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(title_text=f"{FieldType[0]}<sub>y</sub> {PlotType}", title_x=0.5)
    fig.update_layout(font=dict(size=32, family="American Typewriter", color="Black"))
    fig.update_layout(coloraxis_showscale=False)
    fig.update_yaxes(showticklabels=False)
    return fig

@callback(
    Output('Ez-Truth-Graph', 'figure'),
    Input('FieldData', 'data'),
    Input('Field Type', 'value'),
    Input('Plot Type', 'value')
)
def update_figure(data, FieldType, PlotType):
    A = DataSelectionTruth(data, FieldType, PlotType, "Z")
    if PlotType == "Phase":
        colormap = "edge"
    elif PlotType == "Magnitude":
        colormap = "jet"
    if FieldType == "Magnetic":
        FieldType = "Hagnetic"
    fig = px.imshow(A, color_continuous_scale=colormap)
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(title_text=f"{FieldType[0]}<sub>z</sub> {PlotType}", title_x=0.5)
    fig.update_layout(font=dict(size=32, family="American Typewriter", color="Black"))
    fig.update_layout(coloraxis_showscale=False)
    fig.update_yaxes(showticklabels=False)
    return fig

@callback(
    Output('Ex-CNN-Graph', 'figure'),
    Input('FieldData', 'data'),
    Input('Field Type', 'value'),
    Input('Plot Type', 'value')
)
def update_figure(data, FieldType, PlotType):
    A = DataSelectionPrediction(data, FieldType, PlotType, "X")
    if PlotType == "Phase":
        colormap = "edge"
    elif PlotType == "Magnitude":
        colormap = "jet"
    if FieldType == "Magnetic":
        FieldType = "Hagnetic"
    fig = px.imshow(A, color_continuous_scale=colormap, labels = {"y": "CNN"})
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(title_text=f"{FieldType[0]}<sub>x</sub> {PlotType}", title_x=0.5)
    fig.update_layout(font=dict(size=32, family="American Typewriter", color="Black"))
    fig.update_layout(coloraxis_showscale=False)
    fig.update_yaxes(showticklabels=False)
    return fig

@callback(
    Output('Ey-CNN-Graph', 'figure'),
    Input('FieldData', 'data'),
    Input('Field Type', 'value'),
    Input('Plot Type', 'value')
)
def update_figure(data, FieldType, PlotType):
    A = DataSelectionPrediction(data, FieldType, PlotType, "Y")
    if PlotType == "Phase":
        colormap = "edge"
    elif PlotType == "Magnitude":
        colormap = "jet"
    if FieldType == "Magnetic":
        FieldType = "Hagnetic"
    fig = px.imshow(A, color_continuous_scale=colormap)
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(title_text=f"{FieldType[0]}<sub>y</sub> {PlotType}", title_x=0.5)
    fig.update_layout(font=dict(size=32, family="American Typewriter", color="Black"))
    fig.update_layout(coloraxis_showscale=False)
    fig.update_yaxes(showticklabels=False)
    return fig

@callback(
    Output('Ez-CNN-Graph', 'figure'),
    Input('FieldData', 'data'),
    Input('Field Type', 'value'),
    Input('Plot Type', 'value')
)
def update_figure(data, FieldType, PlotType):
    A = DataSelectionPrediction(data, FieldType, PlotType, "Z")
    if PlotType == "Phase":
        colormap = "edge"
    elif PlotType == "Magnitude":
        colormap = "jet"
    if FieldType == "Magnetic":
        FieldType = "Hagnetic"
    fig = px.imshow(A, color_continuous_scale=colormap)
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(title_text=f"{FieldType[0]}<sub>z</sub> {PlotType}", title_x=0.5)
    fig.update_layout(font=dict(size=32, family="American Typewriter", color="Black"))
    fig.update_layout(coloraxis_showscale=False)
    fig.update_yaxes(showticklabels=False)
    return fig




if __name__ == '__main__':
    app.run(debug=True)