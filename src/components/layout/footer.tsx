export function Footer() {
  const year = new Date().getFullYear();
  return (
    <div className="border-t bg-white p-4 text-center text-sm text-muted-foreground">
      (c) {year}{" "}
      <a
        href="https://info.dengue.mat.br"
        target="_blank"
        rel="noopener noreferrer"
        className="underline"
      >
        Infodengue
      </a>{" "}
      All rights reserved. Powered by{" "}
      <a
        href="https://mosqlimate.org"
        target="_blank"
        rel="noopener noreferrer"
        className="underline"
      >
        Mosqlimate
      </a>{" "}
      &amp;{" "}
      <a
        href="https://epigraphhub.org"
        target="_blank"
        rel="noopener noreferrer"
        className="underline"
      >
        EpiGraphHub
      </a>
    </div>
  );
}
