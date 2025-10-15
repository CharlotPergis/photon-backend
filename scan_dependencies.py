import ast
import os

def scan_python_files():
    """Scan all Python files in the project for imports"""
    all_imports = set()
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py') and not file.startswith('scan_'):
                python_files.append(os.path.join(root, file))
    
    print(f"🔍 Scanning {len(python_files)} Python files...")
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            file_imports = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        file_imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        file_imports.add(node.module.split('.')[0])
            
            if file_imports:
                print(f"\n📄 {file_path}:")
                for imp in sorted(file_imports):
                    print(f"   - {imp}")
                    all_imports.add(imp)
                    
        except Exception as e:
            print(f"   ⚠️ Could not parse {file_path}: {e}")
    
    return all_imports

# Standard library modules (don't include)
stdlib = {
    'os', 'io', 're', 'base64', 'datetime', 'timezone', 'requests', 'json',
    'sys', 'uuid', 'threading', 'ssl', 'numpy', 'ast', 'time'
}

all_imports = scan_python_files()
external = all_imports - stdlib

print(f"\n🎯 ALL EXTERNAL PACKAGES NEEDED:")
for pkg in sorted(external):
    print(f"  {pkg}")