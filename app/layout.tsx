import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Skylark BI Agent",
  description: "Auditable business intelligence over live monday.com operations data"
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
