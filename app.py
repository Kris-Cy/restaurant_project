from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.optimize import curve_fit
import pulp

app = Flask(__name__)

# Global storage for points (in a real app, you'd use a database)
page_points = {
    'page1': [],
    'page2': [],
    'page3': [],
    'page4': [],
    'page5': []
}

# Demonstration points for all pages
demo_points = [
    {'x': 1, 'y': 3.5},
    {'x': 2, 'y': 6.2},
    {'x': 3, 'y': 8.1},
    {'x': 4, 'y': 9.5},
    {'x': 5, 'y': 10.6},
    {'x': 6, 'y': 11.4},
    {'x': 7, 'y': 12.0},
    {'x': 8, 'y': 12.4},
    {'x': 9, 'y': 12.7},
    {'x': 10, 'y': 13.0}
]

def linear_function(x, a, b):
    """Linear function: y = ax + b"""
    return a * x + b

def quadratic_function(x, a, b, c):
    """Quadratic function: y = ax² + bx + c"""
    return a * x**2 + b * x + c

def exponential_function(x, a, b, c):
    """Exponential function: y = a * exp(bx) + c"""
    return a * np.exp(b * x) + c

def logistic_function(x, L, k, x0):
    """Logistic function: y = L / (1 + exp(-k(x - x0)))"""
    return L / (1 + np.exp(-k * (x - x0)))

def fit_curve(x_data, y_data):
    """
    Fit multiple curve types and return the best fit
    """
    if len(x_data) < 2:
        return None, None, "Need at least 2 points for curve fitting"
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    best_fit = None
    best_r_squared = -float('inf')
    best_function_name = ""
    
    # Try linear fit
    try:
        popt, _ = curve_fit(linear_function, x_data, y_data)
        y_pred = linear_function(x_data, *popt)
        r_squared = 1 - np.sum((y_data - y_pred) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
        
        if r_squared > best_r_squared:
            best_r_squared = r_squared
            best_fit = {'type': 'linear', 'params': popt.tolist(), 'r_squared': float(r_squared)}
            best_function_name = "Linear"
    except:
        pass
    
    # Try quadratic fit
    try:
        popt, _ = curve_fit(quadratic_function, x_data, y_data)
        y_pred = quadratic_function(x_data, *popt)
        r_squared = 1 - np.sum((y_data - y_pred) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
        
        if r_squared > best_r_squared:
            best_r_squared = r_squared
            best_fit = {'type': 'quadratic', 'params': popt.tolist(), 'r_squared': float(r_squared)}
            best_function_name = "Quadratic"
    except:
        pass
    
    # Try exponential fit (with bounds to prevent overflow)
    try:
        popt, _ = curve_fit(exponential_function, x_data, y_data, 
                          bounds=([-100, -10, -100], [100, 10, 100]))
        y_pred = exponential_function(x_data, *popt)
        r_squared = 1 - np.sum((y_data - y_pred) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
        
        if r_squared > best_r_squared:
            best_r_squared = r_squared
            best_fit = {'type': 'exponential', 'params': popt.tolist(), 'r_squared': float(r_squared)}
            best_function_name = "Exponential"
    except:
        pass
    
    # Try logistic fit (with bounds to ensure reasonable parameters)
    try:
        # Estimate initial parameters
        y_min, y_max = min(y_data), max(y_data)
        L_guess = y_max * 1.1  # Carrying capacity slightly above max
        k_guess = 1.0  # Growth rate
        x0_guess = np.mean(x_data)  # Midpoint
        
        popt, _ = curve_fit(logistic_function, x_data, y_data, 
                          p0=[L_guess, k_guess, x0_guess],
                          bounds=([y_max*0.5, 0.1, min(x_data)], [y_max*2, 10, max(x_data)]))
        y_pred = logistic_function(x_data, *popt)
        r_squared = 1 - np.sum((y_data - y_pred) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
        
        if r_squared > best_r_squared:
            best_r_squared = r_squared
            best_fit = {'type': 'logistic', 'params': popt.tolist(), 'r_squared': float(r_squared)}
            best_function_name = "Logistic"
    except:
        pass
    
    if best_fit is None:
        return None, None, "Could not fit any curve to the data"
    
    # Generate prediction line points within the existing chart boundaries (with 40% padding)
    x_min, x_max = min(x_data), max(x_data)
    x_range = x_max - x_min
    
    # Use the same 40% padding that the frontend applies to chart scales
    x_pred_min = max(0, x_min - x_range * 0.4)  # Don't go below 0
    x_pred_max = x_max + x_range * 0.4
    
    x_pred = np.linspace(x_pred_min, x_pred_max, 200)  # More points for smoother line
    
    if best_fit['type'] == 'linear':
        y_pred = linear_function(x_pred, *best_fit['params'])
    elif best_fit['type'] == 'quadratic':
        y_pred = quadratic_function(x_pred, *best_fit['params'])
    elif best_fit['type'] == 'exponential':
        y_pred = exponential_function(x_pred, *best_fit['params'])
    elif best_fit['type'] == 'logistic':
        y_pred = logistic_function(x_pred, *best_fit['params'])
    
    # Convert numpy arrays to regular Python lists for JSON serialization
    prediction_line = [{'x': float(x), 'y': float(y)} for x, y in zip(x_pred, y_pred)]
    
    return prediction_line, best_fit, f"Best fit: {best_function_name} (R² = {best_r_squared:.3f})"

def regress(points):
    """
    Process points and return regression results with curve fitting.
    """
    if not points:
        return []
    
    # Extract x and y coordinates
    x_coords = [point['x'] for point in points]
    y_coords = [point['y'] for point in points]
    
    # Fit curve to the data
    prediction_line, fit_info, fit_message = fit_curve(x_coords, y_coords)
    
    # Add prediction line to the result
    result = {
        'points': points,  # Original points (no transformation applied)
        'prediction_line': prediction_line,
        'fit_info': fit_info,
        'fit_message': fit_message
    }
    
    return result

@app.route('/')
def home():
    return render_template('page5.html', title='Dish Price Calculator')

@app.route('/page1')
def page1():
    # Load demo points if page is empty
    if not page_points.get('page1'):
        page_points['page1'] = demo_points.copy()
    return render_template('page1.html', title='Optimal Table Turnover', points=page_points.get('page1', []))

@app.route('/page2')
def page2():
    # Load demo points if page is empty
    if not page_points.get('page2'):
        page_points['page2'] = demo_points.copy()
    return render_template('page2.html', title='Price Sensitivity', points=page_points.get('page2', []))

@app.route('/page3')
def page3():
    # Load demo points if page is empty
    if not page_points.get('page3'):
        page_points['page3'] = demo_points.copy()
    return render_template('page3.html', title='Optimal Advertising Budget', points=page_points.get('page3', []))

@app.route('/page4')
def page4():
    # Load demo points if page is empty
    if not page_points.get('page4'):
        page_points['page4'] = demo_points.copy()
    return render_template('page4.html', title='Max Acceptable Wait Time', points=page_points.get('page4', []))

@app.route('/page5')
def page5():
    # Load demo points if page is empty
    if not page_points.get('page5'):
        page_points['page5'] = demo_points.copy()
    return render_template('page5.html', title='Page 5', points=page_points.get('page5', []))

@app.route('/page6')
def page6():
    return render_template('page6.html', title='Employee Efficiency Calculator')

@app.route('/submit_points', methods=['POST'])
def submit_points():
    """Handle point submission from frontend"""
    data = request.get_json()
    page = data.get('page')
    points = data.get('points', [])
    
    if page in page_points:
        page_points[page] = points
        
        # Process points through regress function
        result = regress(points)
        
        return jsonify({
            'success': True,
            'points': result['points'],
            'prediction_line': result['prediction_line'],
            'fit_info': result['fit_info'],
            'fit_message': result['fit_message'],
            'message': f'Processed {len(points)} points for {page}'
        })
    
    return jsonify({
        'success': False,
        'message': 'Invalid page'
    }), 400

@app.route('/load_demo', methods=['POST'])
def load_demo():
    """Load demonstration points for a specific page"""
    data = request.get_json()
    page = data.get('page')
    
    if page in page_points:
        page_points[page] = demo_points.copy()
        
        return jsonify({
            'success': True,
            'points': demo_points,
            'message': f'Loaded {len(demo_points)} demonstration points for {page}'
        })
    
    return jsonify({
        'success': False,
        'message': 'Invalid page'
    }), 400

@app.route('/api/optimize_schedule', methods=['POST'])
def optimize_schedule():
    data = request.get_json()
    employees = data['employees']  # List of dicts: {id, name, receptionist_efficiency, waiter_efficiency, cook_efficiency, cashier_efficiency, cleaner_efficiency}
    days = data['days']            # List of day names, e.g. ['Sun', 'Mon', ...]
    roles = data['roles']          # List of role dicts: {key, label}
    role_limits = data['role_limits']  # Dict: {role_label: limit}
    max_days_per_employee = data.get('max_days_per_employee', 5)
    max_employees_per_day = data.get('max_employees_per_day', 28)

    num_employees = len(employees)
    num_days = len(days)
    num_roles = len(roles)

    # Build efficiency matrix: [employee][day][role]
    eff = [[[0 for _ in range(num_roles)] for _ in range(num_days)] for _ in range(num_employees)]
    for i, emp in enumerate(employees):
        for r, role in enumerate(roles):
            key = role['key']
            for d in range(num_days):
                eff[i][d][r] = emp[key]

    # ILP variables: x[i][d][r] = 1 if employee i is assigned to role r on day d
    x = [[[pulp.LpVariable(f'x_{i}_{d}_{r}', cat='Binary') for r in range(num_roles)] for d in range(num_days)] for i in range(num_employees)]

    prob = pulp.LpProblem('ScheduleOptimization', pulp.LpMaximize)

    # Objective: maximize total efficiency
    prob += pulp.lpSum(eff[i][d][r] * x[i][d][r] for i in range(num_employees) for d in range(num_days) for r in range(num_roles))

    # Constraint: Each employee works at most max_days_per_employee days
    for i in range(num_employees):
        prob += pulp.lpSum(x[i][d][r] for d in range(num_days) for r in range(num_roles)) <= max_days_per_employee

    # Constraint: Each employee can have at most one role per day
    for i in range(num_employees):
        for d in range(num_days):
            prob += pulp.lpSum(x[i][d][r] for r in range(num_roles)) <= 1

    # Constraint: Each role has at most role_limits[role] employees per day (except Waiter, which soaks up the rest)
    for r, role in enumerate(roles):
        if role['label'] == 'Waiter':
            continue  # Waiter will be filled by the total employees per day constraint
        limit = role_limits.get(role['label'], max_employees_per_day)
        for d in range(num_days):
            prob += pulp.lpSum(x[i][d][r] for i in range(num_employees)) <= limit

    # Constraint: No more than and no less than max_employees_per_day employees per day
    for d in range(num_days):
        prob += pulp.lpSum(x[i][d][r] for i in range(num_employees) for r in range(num_roles)) == max_employees_per_day

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Build result: for each day, for each role, list assigned employees
    schedule = {day: {role['label']: [] for role in roles} for day in days}
    for i, emp in enumerate(employees):
        for d, day in enumerate(days):
            for r, role in enumerate(roles):
                if pulp.value(x[i][d][r]) == 1:
                    schedule[day][role['label']].append(emp['name'])

    # Also return the total efficiency
    total_efficiency = int(pulp.value(prob.objective))

    return jsonify({'schedule': schedule, 'total_efficiency': total_efficiency})

if __name__ == '__main__':
    app.run(debug=True) 