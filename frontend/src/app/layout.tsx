import type { Metadata } from "next";
import "./globals.css";
import CookieConsent from "@/components/CookieConsent";
import CursorDotTrail from "@/components/CursorDotTrail";

export const metadata: Metadata = {
  title: "SatQuery AI",
  description: "Multimodal Remote Sensing Image Analysis through Text Queries",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        {children}
        <CookieConsent />
        <CursorDotTrail
          color="#00f0ff"
          colorInverted="#000000"
        />
      </body>
    </html>
  );
}
