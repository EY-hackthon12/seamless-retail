"use client";
import ColorBends from "@/components/ColorBends";
import BlurText from "@/components/BlurText";
import BubbleMenu from "@/components/BubbleMenu";
import ClickSpark from "@/components/ClickSpark";
import { Doto } from "next/font/google";

const doto = Doto({
  subsets: ["latin"],
  weight: ["700"], // available: 400, 500, 700
  display: "swap",
});

const items = [
  {
    label: 'Open LA²',
    href: '/la',
    ariaLabel: 'Home',
    rotation: -8,
    hoverStyles: { bgColor: '#3b82f6', textColor: '#ffffff' }
  },
  {
    label: 'about',
    href: '/about',
    ariaLabel: 'About',
    rotation: 8,
    hoverStyles: { bgColor: '#8b5cf6', textColor: '#ffffff' }
  },
  {
    label: 'contact',
    href: '/contact',
    ariaLabel: 'Contact',
    rotation: 5,
    // hoverStyles: { bgColor: '#8b5cf6', textColor: '#ffffff' }
    hoverStyles: { bgColor: '#10b981', textColor: '#ffffff' }
  }
];


export default function Home() {

  const handleAnimationComplete = () => {
    console.log('Animation completed!');
  };

  return (
    <main className="relative flex items-center justify-center h-screen w-full overflow-hidden">
      {/* 🔥 Global click spark effect (behind all content) */}
      <ClickSpark
        sparkColor="#fff"
        sparkSize={10}
        sparkRadius={15}
        sparkCount={8}
        duration={400}
      />
      {/* <div className="absolute inset-0 bg-gradient-to-b from-purple-900 via-white to-white -z-1"> */}
      <div className="absolute inset-0 bg-gradient-to-b from-purple-900 via-black to-black -z-1">
        <ColorBends
          colors={["#ffffff", "#ff00ff", "#00ffff"]}
          rotation={0}
          speed={0.3}
          scale={1}
          frequency={1}
          warpStrength={1}
          mouseInfluence={0.1}
          parallax={0.6}
          noise={0.08}
          transparent
        />

      </div>

      <div className="relative p-3 flex flex-col items-center justify-center z-2">
        <div className="absolute bg-black opacity-85 rounded-4xl h-full w-full blur-3xl">
        </div>
        <BlurText
          text="Debenture"
          delay={125}
          animateBy="words"
          direction="top"
          onAnimationComplete={handleAnimationComplete}
          className={`text-7xl md:text-8xl font-bold mb-8 text-white ${doto.className}`}
        />
        <BlurText
          text="Agentic AI for Seamless Retail"
          delay={80}
          animateBy="letters"
          direction="top"
          onAnimationComplete={handleAnimationComplete}
          className="text-2xl mb-8 text-white"
        />

        <div className="z-10 mt-8 w-[300px] rounded-3xl flex items-center justify-center p-4">
          <BubbleMenu
            items={[
              {
                label: 'Open SR²',
                href: '/la',
                ariaLabel: 'Open Agent',
                rotation: -8,
                hoverStyles: { bgColor: '#3b82f6', textColor: '#ffffff' }
              },
              {
                label: 'about',
                href: '/about',
                ariaLabel: 'About',
                rotation: 8,
                hoverStyles: { bgColor: '#8b5cf6', textColor: '#ffffff' }
              },
              {
                label: 'contact',
                href: '/contact',
                ariaLabel: 'Contact',
                rotation: 5,
                hoverStyles: { bgColor: '#10b981', textColor: '#ffffff' }
              }
            ]}
            menuBg="#ffffff"
            menuContentColor="#111111"
            animationDuration={0.6}
            staggerDelay={0.15}
          />
        </div>
      </div>
      {/* </ClickSpark> */}
    </main>
  );
}
