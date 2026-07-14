import Image from "next/image";

interface HeaderProps {
  disease: string;
  state: string;
  cases: number;
  year: number;
}

const STATES: Record<string, string> = {
  AC: "Acre", AL: "Alagoas", AM: "Amazonas", AP: "Amapá",
  BA: "Bahia", CE: "Ceará", DF: "Distrito Federal", ES: "Espírito Santo",
  GO: "Goiás", MA: "Maranhão", MG: "Minas Gerais", MS: "Mato Grosso do Sul",
  MT: "Mato Grosso", PA: "Pará", PB: "Paraíba", PE: "Pernambuco",
  PI: "Piauí", PR: "Paraná", RJ: "Rio de Janeiro", RN: "Rio Grande do Norte",
  RO: "Rondônia", RR: "Roraima", RS: "Rio Grande do Sul",
  SC: "Santa Catarina", SE: "Sergipe", SP: "São Paulo", TO: "Tocantins",
};

export function Header({ disease, state, cases, year }: HeaderProps) {
  const diseaseName = disease === "chikungunya" ? "Chikungunya" : disease;
  const stateName = STATES[state] || state;

  return (
    <div className="flex items-center justify-between border-b bg-white p-4">
      <div className="flex items-center gap-3">
        <Image
          src="https://info.dengue.mat.br/static/img/info-dengue-logo-multicidades.png"
          alt="InfoDengue"
          width={40}
          height={40}
        />
        <div>
          <h1 className="text-lg font-bold">Real-time Epidemic Scanner</h1>
          <p className="text-sm text-muted-foreground">Real-time epidemiology</p>
        </div>
      </div>
      <div className="text-right">
        <h2 className="text-base font-semibold">
          Epidemiological Report for {diseaseName}
        </h2>
        <p className="text-sm text-muted-foreground">{stateName}</p>
        <p className="text-sm">
          Cumulative notified cases since Jan {year}:{" "}
          <span className="font-bold">{cases.toLocaleString()}</span>
        </p>
      </div>
    </div>
  );
}
