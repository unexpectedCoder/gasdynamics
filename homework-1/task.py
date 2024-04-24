import csv
import yaml
from dataclasses import dataclass
from math import degrees, radians
from typing import Iterable


@dataclass(frozen=True)
class Task:
    """Класс, хранящий исходные данные задачи.

    Поля
    ----
    variant
        Номер варианта.
    p0
        Давление в камере двигателя, Па.
    T0
        Температура в камере двигателя, К.
    R
        Газовая постоянная, Дж/(кг К).
    k
        Показатель адиабаты.
    d_critic
        Диаметр критического сечения сопла, м.
    area_ratio
        Отношение площади выходного сечения к площади критического.
    d_chamber
        Диаметр камеры двигателя, м.
    alpha
        Угол конуса конфузора, рад.
    beta
        Угол конуса диффузора, рад.
    rel_propel_mass
        Относительная величина запаса топлива.
    """
    variant: float
    p0: float
    T0: float
    R: float
    k: float
    d_critic: float
    area_ratio: float
    d_chamber: float
    alpha: float
    beta: float
    rel_propel_mass: float

    @classmethod
    def from_file(cls, path: str, variant: int | Iterable[int]):
        """Инициализация объекта задачи по переданному файлу с данными.

        На входе
        --------
        path
            Путь к файлу.
        variant
            Номер варианта.

        На выходе
        ---------
            Объект задачи.

        Исключения
        ----------
        ValueError
            Если не поддерживается тип файла данных.
        """
        if path.endswith(".csv"):
            return cls.from_csv(path, variant)
        if path.endswith(".yaml"):
            return cls.from_yaml(path, variant)
        raise ValueError(
            f"неизвестный тип файла с исходными данными `{path}`"
        )
    
    @classmethod
    def from_yaml(cls, path: str, variant: int | Iterable[int]):
        """Чтение данных из файла YAML.

        На входе
        --------
        path
            Путь к файлу.
        variant
            Номер варианта.

        На выходе
        ---------
            Вариант задачи.
        """
        if not isinstance(variant, int):
            with open(path, "r") as f:
                data = [yaml.safe_load(f)[vi] for vi in variant]
                for di in data:
                    di["alpha"] = radians(di["alpha"])
                    di["beta"] = radians(di["beta"])
                return [cls(vi, **di) for vi, di in zip(variant, data)]
        with open(path, "r") as f:
            data = yaml.safe_load(f)[variant]
            data["alpha"] = radians(data["alpha"])
            data["beta"] = radians(data["beta"])
            data["d_chamber"] *= data["d_critic"]
            return cls(variant, **data)
        
    @classmethod
    def from_csv(cls, path: str, variant: int | Iterable[int] | None):
        """Чтение данных из файла CSV.

        На входе
        --------
        path
            Путь к файлу.
        variant
            Номер варианта.

        На выходе
        ---------
            Вариант задачи.
        """
        if isinstance(variant, Iterable):
            variant = list(variant)
            tasks = []
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    var = int(row["variant"])
                    if var in variant:
                        data = {k: float(v) for k,v in row.items()}
                        data["variant"] = var
                        data["alpha"] = radians(data["alpha"])
                        data["beta"] = radians(data["beta"])
                        data["d_chamber"] *= data["d_critic"]
                        tasks.append(cls(**data))
                        variant.remove(var)
                        if variant == []:
                            break
                return tasks
        elif variant is None:
            tasks = []
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    var = int(row["variant"])
                    data = {k: float(v) for k,v in row.items()}
                    data["variant"] = var
                    data["alpha"] = radians(data["alpha"])
                    data["beta"] = radians(data["beta"])
                    data["d_chamber"] *= data["d_critic"]
                    tasks.append(cls(**data))
                return tasks
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                var = int(row["variant"])
                if var == variant:
                    data = {k: float(v) for k,v in row.items()}
                    data["variant"] = var
                    data["alpha"] = radians(data["alpha"])
                    data["beta"] = radians(data["beta"])
                    data["d_chamber"] *= data["d_critic"]
                    return cls(**data)
        raise ValueError(
            f"вариант {variant} не найден"
        )
    
    def as_tuple(self):
        """Представление класса в виде кортежа.

        На выходе
        ---------
            Кортеж с данными класса.
        """
        return (
            self.variant,
            self.p0,
            self.T0,
            self.R,
            self.k,
            self.d_critic,
            self.area_ratio,
            self.d_chamber,
            degrees(self.alpha),
            degrees(self.beta),
            self.rel_propel_mass
        )
