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
      .{' '}
      <a
        href="https://royalsocietypublishing.org/rsos/article/12/5/241261/235685/Large-scale-epidemiological-modelling-scanning-for"
        target="_blank"
        rel="noopener noreferrer"
        className="underline"
      >
        EpiScanner Article
      </a>
    </div>
  );
}
