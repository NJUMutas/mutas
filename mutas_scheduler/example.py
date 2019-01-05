from mutas_scheduler.core.mutas_solver import MutasSolver
from mutas_scheduler.data.global_data import GlobalData

if __name__ == '__main__':
    ms = MutasSolver()
    gd = GlobalData()
    ms.solve(gd)
    print(ms.a)
    print(ms.f)