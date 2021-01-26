# Finite Element Program by Pierre Kerfriden and Ehsan Mikaeili - Mines ParisTech and Cardiff University
# brain model by Remigiusz Krepczynski
# https://grabcad.com/library/brain-the-brain

from dolfin import *
import numpy as np
import os

class void_IO:
    def __init__(self):
        pass

def brain_shift( IO = void_IO() ):

    # Optimization options for the form compiler
    parameters["form_compiler"]["cpp_optimize"] = True
    ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

    
    E1 = 10.0
    if hasattr(IO, 'elastic_modulus_sphere'):
        E0 = pow(10, IO.input_FEM['Elastic Modulus'])
    else:
        E0 = 2.

    if hasattr(IO, 'radius_incision'):
        radius_incision = IO.input_FEM['Incision Radius']
    else:
        radius_incision = 1.

    if hasattr(IO, 'position_sphere_x'):
        center_x = IO.position_sphere_x
    else:
        center_x = 0.5
    if hasattr(IO, 'position_sphere_y'):
        center_y = IO.position_sphere_y
    else:
        center_y = 0.5
    if hasattr(IO, 'position_sphere_y'):
        center_y = IO.position_sphere_y
    else:
        center_z = 1.5
    center = Point(center_x,center_y,center_z)

    t0 = 100.
    t1 = 0.01

    radius = 0.5

    # Meshing
    os.system('dolfin-convert brain_finer.msh mesh.xml')
    mesh = Mesh("mesh.xml")
    n = FacetNormal(mesh)

    file_mesh = File("results/mesh.pvd")
    file_mesh  << mesh

    V = VectorFunctionSpace(mesh, 'P', 1)
    W = FunctionSpace(mesh, 'P', 1)

    # Define boundary condition
    class incision_func(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and ( (x[2] >= -0.25) and ( ((x[0]-center[0])**2+(x[1]-center[1])**2) < radius_incision**2 ) )

    incision = incision_func()

    boundary_marker = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_marker.set_all(0)
    incision.mark(boundary_marker, 1)

    file_boundary_marker = File("results/boundary_marker.pvd")
    file_boundary_marker << boundary_marker

    ds =Measure('ds', domain=mesh, subdomain_data=boundary_marker)

    bcs = []

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    class tumor(UserExpression):
        def eval(self, value, x):
            smoothing_length = 0.01 * radius
            distance = sqrt( pow(x[0]-center[0],2) + pow(x[1]-center[1],2) + pow(x[2]-center[2],2) ) - radius
            alpha = sigmoid( distance / smoothing_length )
            value[0] = 0. * alpha + 1. * (1-alpha)

    class elasticity_field(UserExpression):
        def eval(self, value, x):
            smoothing_length = 0.1 * radius
            distance = sqrt( pow(x[0]-center[0],2) + pow(x[1]-center[1],2) + pow(x[2]-center[2],2) ) - radius
            alpha = sigmoid( distance / smoothing_length )
            value[0] = np.exp( np.log(E0) * alpha + np.log(E1) * (1-alpha) )

    class surface_elasticity_field(UserExpression):
        def eval(self, value, x):
            smoothing_length = 0.25 * radius_incision
            distance = sqrt( pow(x[0]-center[0],2) + pow(x[1]-center[1],2) + pow(x[2]-2.5,2) ) - radius_incision
            alpha = sigmoid( distance / smoothing_length )
            value[0] = np.exp( np.log(t0) * alpha + np.log(t1) * (1-alpha) )
    
    T = tumor(element=W.ufl_element())
    tumor_interp = Function(W)
    tumor_interp.interpolate(T)
    tumor_interp.rename('tumor','tumor')
    file_elasticity_coefficient = File("results/tumor.pvd")
    file_elasticity_coefficient << tumor_interp

    E = elasticity_field(element=W.ufl_element())
    elasticity_field_interp = Function(W)
    elasticity_field_interp.interpolate(E)
    elasticity_field_interp.rename('elasticity_field','elasticity_field')
    file_elasticity_coefficient = File("results/elasticity_coefficient.pvd")
    file_elasticity_coefficient << elasticity_field_interp

    k = surface_elasticity_field(element=W.ufl_element())
    elasticity_field_interp = Function(W)
    elasticity_field_interp.interpolate(k)
    elasticity_field_interp.rename('elasticity_field','elasticity_field')
    file_elasticity_coefficient = File("results/surface_elasticity_coefficient.pvd")
    file_elasticity_coefficient << elasticity_field_interp

    B = Expression(('0.0','0.0','-1.0*t' ),t=0,degree=1)    # Body force per unit volume
    T  = Constant((0.,  0.0 , 0.))                          # Traction force on the boundary

    # Elasticity parameters
    nu = 0.3
    mu, lmbda = E/(2*(1 + nu)), E*nu/((1 + nu)*(1 - 2*nu))

    v  = TestFunction(V)             # Test function
    u  = Function(V)                 # Displacement from previous iteration
    du = TrialFunction(V)            # Incremental displacement

    # Kinematics
    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor

    # Invariants of deformation tensors
    Ic = tr(C)
    J  = det(F)

    # Stored strain energy density (compressible neo-Hookean model)
    psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2 

    # Total potential energy
    Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds + inner(k*u,u)*ds(0) 

    # Compute first variation of Pi (directional derivative about u in the direction of v)
    F = derivative(Pi, u, v)

    # Compute Jacobian of F
    J = derivative(F, u, du)

    # Solve variational problem
    solve(F == 0, u, bcs, J=J,form_compiler_parameters=ffc_options) 

    file_displacement = File("results/displacement.pvd")
    u.rename('displacement','displacement')
    file_displacement << u

    T=1
    dt = 0.25
    t = dt
    time_step = 0
    
    while t < T + DOLFIN_EPS:

        time_step +=1
        print("time_step",time_step)

        B.t = t

        # Solve variational problem
        solve(F == 0, u, bcs, J=J, form_compiler_parameters=ffc_options)
    
        u.rename('displacement','displacement')
        file_displacement << u

        t+=dt
    
    class sensor_field(UserExpression):
        def eval(self, value, x):
            smoothing_length = 0.5
            center_sensor = center
            Num = pow(x[0]-center_sensor[0],2) + pow(x[1]-center_sensor[1],2) + pow(x[2]-center_sensor[2],2)
            normal_const = np.sqrt( pow( (2*np.pi) , 3) * pow( smoothing_length**2 , 3) )
            value[0] = 1./normal_const * np.exp( - 0.5* Num / pow(smoothing_length,2) )

    sensor = sensor_field(element=W.ufl_element())
    sensor_interp = Function(W)
    sensor_interp.interpolate(sensor)
    sensor_interp.rename('sensor','sensor')
    file_sensor = File("results/sensor.pvd")
    file_sensor << sensor_interp

    sensor_disp_form = inner( u , sensor * Constant((1.0,0.0,0.0)) ) * dx
    sensor_disp_x = assemble(sensor_disp_form)

    sensor_disp_form = inner( u , sensor * Constant((0.0,0.0,1.0)) ) * dx
    sensor_disp_z = assemble(sensor_disp_form)

    sensor_disp_form = inner( u , sensor * Constant((0.0,1.0,0.0)) ) * dx
    sensor_disp_y = assemble(sensor_disp_form)
    print("sensor_disp_z",sensor_disp_z)

    vol_sensor_form = sensor * dx(mesh)
    vol_sensor = assemble(vol_sensor_form)
    print("vol_sensor",vol_sensor)

    print("sensor_disp_z/vol_sensor",sensor_disp_z/vol_sensor)

    if hasattr(IO, 'output_FEM'):
        IO.output_FEM ={'displacement_tumor_x': sensor_disp_x/vol_sensor, 'displacement_tumor_y': sensor_disp_y/vol_sensor, 'displacement_tumor_z': sensor_disp_z/vol_sensor}
        
    else:
         print({'displacement_tumor_x': sensor_disp_x/vol_sensor, 'displacement_tumor_y': sensor_disp_y/vol_sensor, 'displacement_tumor_z': sensor_disp_z/vol_sensor})
        
    #if hasattr(IO, 'displacement_tumor_y'):
    #    IO.displacement_tumor_y = sensor_disp_y/vol_sensor
        
    #if hasattr(IO, 'displacement_tumor_z'):
    #    IO.displacement_tumor_z = sensor_disp_z/vol_sensor

