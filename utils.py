def run(args, cwd=None):
    print(subprocess.run(args, stdout=subprocess.PIPE, cwd=cwd).stdout.decode('utf-8'))