"use client"

import { useEffect, useRef, useState } from "react"
import Link from "next/link"
import { Card, CardContent } from "@/components/ui/card"
import { gsap } from "gsap"
import { ChevronLeft, ChevronRight } from "lucide-react"

export default function Home() {
  const textRef = useRef(null)
  const carouselRef = useRef<HTMLDivElement>(null)
  const [currentIndex, setCurrentIndex] = useState(0)
  // Add state for client-side rendering
  const [isClient, setIsClient] = useState(false)

  // Categories data
  const categories = [
    {
      id: "temples",
      title: "Temples",
      description: "Ancient temples with stunning architecture",
      image: "/placeholder.svg?height=300&width=400",
    },
    {
      id: "waterfalls",
      title: "Waterfalls",
      description: "Breathtaking waterfalls amidst lush greenery",
      image: "/placeholder.svg?height=300&width=400",
    },
    {
      id: "forts",
      title: "Forts",
      description: "Historic forts with rich cultural heritage",
      image: "/placeholder.svg?height=300&width=400",
    },
    {
      id: "mountains",
      title: "Mountains",
      description: "Majestic mountains and scenic landscapes",
      image: "/placeholder.svg?height=300&width=400",
    },
    {
      id: "food",
      title: "Food",
      description: "Delicious local cuisine and delicacies",
      image: "/placeholder.svg?height=300&width=400",
    },
    {
      id: "dance-art",
      title: "Dance & Art Forms",
      description: "Traditional dance and art forms",
      image: "/placeholder.svg?height=300&width=400",
    },
    {
      id: "clothes",
      title: "Clothes",
      description: "Traditional attire and textiles",
      image: "/placeholder.svg?height=300&width=400",
    },
  ]

  // Number of cards to show based on screen size
  const getVisibleCards = () => {
    if (typeof window !== "undefined") {
      if (window.innerWidth < 768) return 1
      if (window.innerWidth < 1024) return 2
      return 3
    }
    return 3
  }

  // Use a client-side only value for visibleCards
  const [visibleCards, setVisibleCards] = useState(3)

  // Update visibleCards after component mounts
  useEffect(() => {
    setVisibleCards(getVisibleCards())
    setIsClient(true)
    
    // Add resize listener
    const handleResize = () => {
      setVisibleCards(getVisibleCards())
    }
    
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  const handlePrev = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1)
    }
  }

  const handleNext = () => {
    if (currentIndex < categories.length - visibleCards) {
      setCurrentIndex(currentIndex + 1)
    }
  }

  useEffect(() => {
    // Animate the text overlay when the page loads
    gsap.from(textRef.current, {
      opacity: 0,
      y: 50,
      duration: 1,
      delay: 0.5,
    })

    // Animate cards from right to left
    gsap.from(".card-item", {
      x: 100,
      opacity: 0,
      duration: 0.8,
      stagger: 0.2,
      ease: "power2.out",
    })
  }, [])

  useEffect(() => {
    // Animate carousel slide
    if (carouselRef.current) {
      gsap.to(carouselRef.current, {
        x: -currentIndex * (100 / visibleCards) + "%",
        duration: 0.5,
        ease: "power2.out",
      })
    }
  }, [currentIndex, visibleCards])

  return (
    <main className="flex min-h-screen flex-col items-center bg-gradient-to-b from-amber-50 to-amber-100">
      {/* Header Section with Video Background */}
      <section className="relative w-full h-screen overflow-hidden">
        {/* Use a fallback background color if video/poster fails to load */}
        <div className="absolute top-0 left-0 w-full h-full bg-amber-800"></div>
        
        <video
          autoPlay
          muted
          loop
          playsInline
          className="absolute top-0 left-0 w-full h-full object-cover"
          poster="/video-poster.jpg"
          onError={(e) => {
            // Hide the video element if there's an error loading it
            e.currentTarget.style.display = 'none';
          }}
        >
          <source src="/videos/v.mp4" type="video/mp4" />
          <source src="/videos/karnataka-cinematic-4k.webm" type="video/webm" />
          Your browser does not support the video tag.
        </video>
      
        {/* Semi-transparent overlay */}
        <div className="absolute top-0 left-0 w-full h-full bg-black bg-opacity-50"></div>
      
        {/* Text Overlay */}
        <div
          ref={textRef}
          className="absolute top-0 left-0 w-full h-full flex flex-col items-center justify-center text-white text-center px-4"
        >
          <h1 className="text-4xl md:text-6xl font-bold mb-4 text-amber-300 drop-shadow-lg">
            Welcome to Karnataka Tourism
          </h1>
          <p className="text-xl md:text-3xl font-semibold text-amber-200 drop-shadow-md">
            Karnataka: One State, Many Worlds
          </p>
        </div>
      </section>

      {/* Cards Section with Horizontal Slider */}
      <section className="w-full py-16 px-4 bg-gradient-to-r from-amber-900/10 to-amber-700/10">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12 text-amber-900 font-josefin">Explore Karnataka</h2>

          <div className="relative">
            {/* Carousel Navigation Buttons */}
            <button
              onClick={handlePrev}
              className={`absolute left-0 top-1/2 -translate-y-1/2 z-10 bg-amber-800 text-amber-100 rounded-full p-2 shadow-lg transition-all ${
                currentIndex === 0 ? "opacity-50 cursor-not-allowed" : "opacity-100 hover:bg-amber-700"
              }`}
              disabled={currentIndex === 0}
            >
              <ChevronLeft size={24} />
            </button>

            {/* Carousel Container */}
            <div className="overflow-hidden">
              {isClient ? (
                <div
                  ref={carouselRef}
                  className="flex transition-transform duration-500 ease-out"
                  style={{ width: `${(categories.length / visibleCards) * 100}%` }}
                >
                  {categories.map((category, index) => (
                    <div key={category.id} className="card-item px-2" style={{ width: `${100 / categories.length}%` }}>
                      <Link href={`/category/${category.id}`}>
                        <Card className="overflow-hidden transition-all duration-300 hover:shadow-[0_0_15px_rgba(180,130,10,0.5)] hover:scale-105 h-full border-amber-200 bg-gradient-to-b from-amber-50 to-amber-100">
                          <div className="relative h-48 overflow-hidden">
                            <img
                              src={category.image || "/placeholder.svg"}
                              alt={category.title}
                              className="w-full h-full object-cover transition-transform duration-300 hover:scale-110"
                            />
                          </div>
                          <CardContent className="p-4 bg-gradient-to-r from-amber-800/10 to-amber-700/10">
                            <h3 className="text-xl font-bold mb-2 text-amber-900 font-josefin">{category.title}</h3>
                            <p className="text-amber-800">{category.description}</p>
                          </CardContent>
                        </Card>
                      </Link>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex">
                  {/* Simple placeholder during server render */}
                  {categories.slice(0, 3).map((category, index) => (
                    <div key={category.id} className="card-item px-2" style={{ width: '33.333%' }}>
                      <Card className="overflow-hidden h-full border-amber-200 bg-gradient-to-b from-amber-50 to-amber-100">
                        <div className="relative h-48 overflow-hidden">
                          <img
                            src={category.image || "/placeholder.svg"}
                            alt={category.title}
                            className="w-full h-full object-cover transition-transform duration-300 hover:scale-110"
                          />
                        </div>
                        <CardContent className="p-4 bg-gradient-to-r from-amber-800/10 to-amber-700/10">
                          <h3 className="text-xl font-bold mb-2 text-amber-900 font-josefin">{category.title}</h3>
                          <p className="text-amber-800">{category.description}</p>
                        </CardContent>
                      </Card>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <button
              onClick={handleNext}
              className={`absolute right-0 top-1/2 -translate-y-1/2 z-10 bg-amber-800 text-amber-100 rounded-full p-2 shadow-lg transition-all ${
                currentIndex >= categories.length - visibleCards
                  ? "opacity-50 cursor-not-allowed"
                  : "opacity-100 hover:bg-amber-700"
              }`}
              disabled={currentIndex >= categories.length - visibleCards}
            >
              <ChevronRight size={24} />
            </button>
          </div>

          {/* Pagination Dots - Only render on client */}
          {isClient && (
            <div className="flex justify-center mt-6 gap-2">
              {Array.from({ length: categories.length - visibleCards + 1 }).map((_, index) => (
                <button
                  key={index}
                  onClick={() => setCurrentIndex(index)}
                  className={`w-3 h-3 rounded-full transition-all ${
                    currentIndex === index ? "bg-amber-800 w-6" : "bg-amber-400"
                  }`}
                  aria-label={`Go to slide ${index + 1}`}
                />
              ))}
            </div>
          )}
        </div>
      </section>
    </main>
  )
}
