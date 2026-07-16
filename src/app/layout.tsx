import type { Metadata } from "next";
import { Plus_Jakarta_Sans, Geist_Mono } from "next/font/google";
import "./globals.css";

const jakarta = Plus_Jakarta_Sans({
  subsets: ["latin"],
  variable: "--font-jakarta",
  display: "swap",
});

const geistMono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-mono-geist",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Epi Scanner — Real-time Epidemic Scanner",
  description:
    "Real-time epidemiology dashboard for dengue and arbovirus surveillance across Brazilian states.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`bg-background ${jakarta.variable} ${geistMono.variable}`}
    >
      <head>
        <link
          rel="icon"
          href="https://info.dengue.mat.br/static/img/favicon.ico"
        />
      </head>
      <body className="font-sans antialiased">{children}</body>
    </html>
  );
}
