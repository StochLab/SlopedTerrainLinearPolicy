ne
		if(args.WedgeOrientation<0):
			env.incline_ori_anti = False
		else:
			env.incline_ori_anti = True
		env.incline_ori = math.radians(args.WedgeOrientation)
		env.SetFootFriction(args.FrictionCoeff)
		env.SetLinkMass(0,args.FrontMass)
		env.SetLinkMass(11,args.BackMass)
		env.clips = args.MotorStrength

	state = env.reset()
	if(env.incline_ori_anti == True):
		wedgeori = env.incline_ori
	else:
		wedgeori = -env.incline_ori

	print (
	bold(blue("\nTest Parameters:\n")),
	green('\nWedge Inclination:'),red(env.incline_deg),
	green('\nWedge Orientation:'),red(math.degrees(wedgeori)),
	green('\nCoeff. of friction:'),red(env.friction),